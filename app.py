import gradio as gr
import pandas as pd
import onnxruntime
import numpy as np
from transformers import CamembertTokenizer
from pathlib import Path
import json
import torch.nn.functional as F
import torch
from tqdm import tqdm
import tempfile # NOUVEAU : Import pour gérer les fichiers temporaires
import os

# --- CONFIGURATION ---
CONFIG = {
    # !! VÉRIFIEZ BIEN QUE CES NOMS DE FICHIERS SONT CORRECTS !!
    "onnx_model_path": Path("models/product_classifier.onnx"),
    "target_map_path": Path("models/dataset_target_map.json"),
    "model_name": "camembert-base",
    "max_len": 128,
    "batch_size": 32
}

# --- CHARGEMENT DES ARTEFACTS ---
print("Chargement du modèle, du tokenizer et du mapping...")

try:
    with open(CONFIG["target_map_path"], 'r', encoding='utf-8') as f:
        target_map = json.load(f)
    inverse_target_map = {int(v): k for k, v in target_map.items()}
except FileNotFoundError:
    print(f"ERREUR : Fichier de mapping non trouvé à {CONFIG['target_map_path']}.")
    exit()

tokenizer = CamembertTokenizer.from_pretrained(CONFIG["model_name"])

try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(str(CONFIG["onnx_model_path"]), providers=providers)
    print(f"Modèle ONNX chargé avec succès. Utilisation de : {session.get_providers()[0]}")
except Exception as e:
    print(f"ERREUR : Impossible de charger le modèle ONNX depuis {CONFIG['onnx_model_path']}. Détails : {e}")
    exit()

# --- FONCTIONS LOGIQUES ---

def prepare_text_feature(df):
    text_cols = ['product_name_decli', 'brand', 'summary', 'description']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
    df[text_cols] = df[text_cols].fillna('')
    df['text_feature'] = (
        df['product_name_decli'] + ' ' +
        df['brand'] + ' ' +
        df['summary'] + ' ' +
        df['description']
    ).str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

def predict_single(product_name, brand, description):
    if not product_name:
        return {"Erreur": 1.0, "Veuillez entrer un nom de produit": 0.0}

    text_feature = (product_name + ' ' + (brand or '') + ' ' + (description or '')).strip()
    
    encoding = tokenizer.encode_plus(
        text_feature, max_length=CONFIG["max_len"], padding='max_length', truncation=True, return_tensors='np'
    )
    onnx_inputs = {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}
    
    logits = session.run(None, onnx_inputs)[0]
    probabilities = F.softmax(torch.tensor(logits), dim=1).numpy().flatten()
    
    top5_indices = probabilities.argsort()[-5:][::-1]
    top5_results = {inverse_target_map.get(i, "Inconnu"): float(probabilities[i]) for i in top5_indices}
    
    return top5_results

def predict_csv(uploaded_file):
    if uploaded_file is None:
        return None, "Veuillez charger un fichier CSV."

    try:
        # Gradio fournit un chemin temporaire pour le fichier uploadé
        df = pd.read_csv(uploaded_file.name)
    except Exception as e:
        return None, f"Erreur lors de la lecture du fichier : {e}"
        
    df = prepare_text_feature(df)
    texts_to_predict = df['text_feature'].tolist()
    
    all_preds_text = []
    all_confidences = []

    print(f"Traitement de {len(texts_to_predict)} produits...")
    for i in tqdm(range(0, len(texts_to_predict), CONFIG["batch_size"])):
        batch_texts = texts_to_predict[i:i+CONFIG["batch_size"]]
        encoding = tokenizer.batch_encode_plus(
            batch_texts, max_length=CONFIG["max_len"], padding='max_length', truncation=True, return_tensors='np'
        )
        onnx_inputs = {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}
        logits = session.run(None, onnx_inputs)[0]
        probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        confidences = probabilities.max(axis=1)
        predicted_indices = np.argmax(logits, axis=1)
        all_preds_text.extend([inverse_target_map.get(idx, "Inconnu") for idx in predicted_indices])
        all_confidences.extend(confidences)
        
    df['predicted_category'] = all_preds_text
    df['confidence_score'] = all_confidences
    df = df.drop(columns=['text_feature'])
    
    # --- CORRECTION DE LA GESTION DU FICHIER DE SORTIE ---
    # On utilise tempfile pour créer un fichier de sortie compatible tous systèmes
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.csv', encoding='utf-8-sig') as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        # On retourne le chemin du fichier temporaire créé
        return tmp_file.name, "Prédictions terminées avec succès."

# --- CRÉATION DE L'INTERFACE GRADIO ---
with gr.Blocks(title="Classifieur de Produits Vélo", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🚴‍♂️ Classifieur de Catégories de Produits Vélo")
    gr.Markdown("Utilisez cette interface pour prédire la catégorie de produits de vélo, soit interactivement, soit en traitant un fichier CSV complet.")

    with gr.Tabs():
        with gr.TabItem("Prédiction Interactive"):
            with gr.Row():
                with gr.Column(scale=2):
                    product_name_input = gr.Textbox(label="Nom du produit", placeholder="Ex: Maillot de vélo manches courtes...")
                    brand_input = gr.Textbox(label="Marque (optionnel)", placeholder="Ex: Castelli")
                    description_input = gr.Textbox(label="Description (optionnel)", placeholder="Ex: Un maillot très léger et respirant...")
                    submit_btn_single = gr.Button("Prédire la catégorie", variant="primary")
                with gr.Column(scale=1):
                    output_label = gr.Label(num_top_classes=5, label="Top 5 des prédictions")
        
        with gr.TabItem("Prédiction par Fichier (Batch)"):
            with gr.Column():
                gr.Markdown("Chargez un fichier CSV. La sortie sera un nouveau fichier CSV avec les colonnes `predicted_category` et `confidence_score` ajoutées.")
                file_input = gr.File(label="Fichier CSV d'entrée", file_types=[".csv"])
                status_text = gr.Textbox(label="Statut", interactive=False)
                file_output = gr.File(label="Fichier CSV de sortie")
                submit_btn_batch = gr.Button("Lancer la prédiction par fichier", variant="primary")

    submit_btn_single.click(
        fn=predict_single,
        inputs=[product_name_input, brand_input, description_input],
        outputs=output_label
    )
    
    submit_btn_batch.click(
        fn=predict_csv,
        inputs=file_input,
        outputs=[file_output, status_text]
    )


# --- LANCEMENT DE L'APPLICATION ---
if __name__ == "__main__":
    iface.launch()
