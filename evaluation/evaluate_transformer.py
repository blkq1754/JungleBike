import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import CamembertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
.
MODEL_PATH = PROJECT_ROOT / "models" / "product_classifier.onnx" 
TARGET_MAP_PATH = PROJECT_ROOT / "models" / "dataset_target_map.json" 

EVAL_SET_PATH = PROJECT_ROOT / "data" / "evaluation_set.csv" #fichier csv d'évaluation
MODEL_NAME = "camembert-base"
BATCH_SIZE = 16
OUTPUT_DIR = SCRIPT_DIR / "transformer_model_results"


def evaluate_transformer_model():
    print("--- Évaluation du Modèle Transformer (mis à jour avec 'brand') ---")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    
    print(f"Utilisation de l'appareil : {device.upper()}")

    df = pd.read_csv(EVAL_SET_PATH)
    with open(TARGET_MAP_PATH, 'r', encoding='utf-8') as f:
        target_map = json.load(f)
    inverse_target_map = {int(v): k for k, v in target_map.items()} 
    
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    session = onnxruntime.InferenceSession(str(MODEL_PATH), providers=providers)


    text_cols = ['product_name_decli', 'brand', 'summary', 'description']
    for col in text_cols:
        if col not in df.columns: df[col] = ""
    df[text_cols] = df[text_cols].fillna('')
    df['text_feature'] = (df['product_name_decli'] + ' ' + df['brand'] + ' ' + df['summary'] + ' ' + df['description']).str.strip()
    
    texts_to_predict = df['text_feature'].tolist()
    y_true = df['category'].tolist()

    print("Génération des prédictions...")
    all_preds_text = []
    for i in tqdm(range(0, len(texts_to_predict), BATCH_SIZE), desc="Prédiction en batch"):
        batch_texts = texts_to_predict[i:i+BATCH_SIZE]
        encoding = tokenizer.batch_encode_plus(
            batch_texts, max_length=128, padding='max_length', truncation=True, return_tensors='np'
        )
        onnx_inputs = {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}
        logits = session.run(None, onnx_inputs)[0]
        predicted_indices = np.argmax(logits, axis=1)
        all_preds_text.extend([inverse_target_map[idx] for idx in predicted_indices])
    
    y_pred = all_preds_text

    print("Génération du rapport de classification...")
    labels = list(target_map.keys())
    report = classification_report(y_true, y_pred, labels=labels, output_dict=False, zero_division=0)
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"Rapport sauvegardé dans {report_path}")

    print("Génération de la matrice de confusion...")
    fig, ax = plt.subplots(figsize=(40, 40))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation='vertical', labels=labels)
    plt.title("Matrice de Confusion - Modèle Transformer (avec 'brand')")
    plot_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Matrice de confusion sauvegardée dans {plot_path}")
    print("--- Évaluation terminée ---")

if __name__ == "__main__":
    evaluate_transformer_model()