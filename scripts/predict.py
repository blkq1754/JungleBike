import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CamembertTokenizer
from pathlib import Path
import json
import argparse 
from tqdm import tqdm 
from src.model import ProductClassifier

CONFIG = {
    "model_path": Path("models/product_classifier_model.bin"),
    "target_map_path": Path("models/target_map.json"),
    "model_name": "camembert-base",
    "max_len": 128,
    "batch_size": 32 
}

class Predictor:
    def __init__(self, model_path, target_map_path, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with open(target_map_path, 'r', encoding='utf-8') as f:
            self.target_map = json.load(f)
        
        self.inverse_target_map = {v: k for k, v in self.target_map.items()}
        n_classes = len(self.target_map)

        self.model = ProductClassifier(n_classes=n_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.model = self.model.to(self.device)
        self.model.eval() 
        
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)

    def predict_batch(self, texts: list[str]):
        """Fait des prédictions sur un lot de textes."""
        
        # Tokenizer le texte
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=CONFIG["max_len"],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # les prédictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, dim=1)

        # MAPPING
        predicted_categories = [self.inverse_target_map[idx.item()] for idx in predicted_indices]
        
        return predicted_categories, confidences.cpu().numpy()

def run_predictions(input_csv, output_csv):
    """Charge les données, lance les prédictions par lot et sauvegarde les résultats."""
    print("Chargement du modèle et du tokenizer...")
    predictor = Predictor(
        model_path=CONFIG["model_path"],
        target_map_path=CONFIG["target_map_path"],
        model_name=CONFIG["model_name"]
    )
    print("Modèle prêt.")

    print(f"Lecture du fichier d'entrée : {input_csv}")
    df = pd.read_csv(input_csv)
    text_cols = ['product_name_decli', 'summary', 'description']
    for col in text_cols:
        if col not in df.columns:
            df[col] = "" 
    df[text_cols] = df[text_cols].fillna('')
    df['text_feature'] = (
        df['product_name_decli'] + ' ' +
        df['summary'] + ' ' +
        df['description']
    ).str.strip().str.replace(r'\s+', ' ', regex=True)

    texts_to_predict = df['text_feature'].tolist()
    
    results = []
    confidences = []

    for i in tqdm(range(0, len(texts_to_predict), CONFIG["batch_size"])):
        batch_texts = texts_to_predict[i:i+CONFIG["batch_size"]]
        
        batch_predictions, batch_confidences = predictor.predict_batch(batch_texts)
        
        results.extend(batch_predictions)
        confidences.extend(batch_confidences)

    df['predicted_category'] = results
    df['confidence_score'] = confidences
    df = df.drop(columns=['text_feature'])

    print(f"Sauvegarde des résultats dans : {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig') 
    print("Prédictions terminées avec succès.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire les catégories de produits à partir d'un fichier CSV.")
    parser.add_argument("input_file", type=str, help="Chemin vers le fichier CSV d'entrée.")
    parser.add_argument("output_file", type=str, help="Chemin vers le fichier CSV de sortie.")
    
    args = parser.parse_args()
    
    run_predictions(args.input_file, args.output_file)