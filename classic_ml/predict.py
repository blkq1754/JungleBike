import pandas as pd
import joblib
import argparse
from tqdm import tqdm
from . import config
from .data import load_data

def run_batch_prediction(input_file: str, output_file: str):
    """
    Charge le modèle classique, prédit les catégories pour un fichier CSV d'entrée,
    et sauvegarde les résultats dans un nouveau fichier CSV.
    """
    # Vérifier si le modèle existe
    if not config.MODEL_PATH.exists():
        print(f"Erreur : Modèle non trouvé à l'emplacement : {config.MODEL_PATH}")
        print("Veuillez d'abord entraîner le modèle avec 'classic_ml/train.py'")
        return

    # Charger le pipeline entraîné
    print("Chargement du modèle classique...")
    model_pipeline = joblib.load(config.MODEL_PATH)
    print("Modèle chargé.")

    # Charger les données d'entrée.
    print(f"Lecture du fichier d'entrée : {input_file}")
    df = pd.read_csv(input_file)
    
    # Préparer la feature textuelle unifiée
    text_cols = ['product_name_decli', 'brand', 'summary', 'description']
    for col in text_cols:
        if col not in df.columns:
            df[col] = "" # Ajouter les colonnes si elles manquent
            
    df[text_cols] = df[text_cols].fillna('')
    
    df['text_feature'] = (
        df['product_name_decli'] + ' ' +
        df['brand'] + ' ' +
        df['summary'] + ' ' +
        df['description']
    ).str.strip().str.replace(r'\s+', ' ', regex=True)
    
    texts_to_predict = df['text_feature']

    #les prédictions
    if not texts_to_predict.empty:
        print("Prédiction des catégories en cours...")
        predictions = model_pipeline.predict(texts_to_predict)
        
        print("Calcul des probabilités de confiance...")
        probabilities = model_pipeline.predict_proba(texts_to_predict)
        confidences = probabilities.max(axis=1) 

        # Ajout des resultats
        df['predicted_category'] = predictions
        df['confidence_score'] = confidences

    # supprimer la colonne textuelle unifiée
    df = df.drop(columns=['text_feature'])
    
    # Save 
    print(f"Sauvegarde des résultats dans : {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig') 
    print("Prédictions terminées avec succès.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire les catégories de produits avec le modèle classique (TF-IDF + Régression Logistique).")
    parser.add_argument("input_file", type=str, help="Chemin vers le fichier CSV d'entrée.")
    parser.add_argument("output_file", type=str, help="Chemin vers le fichier CSV de sortie.")
    
    args = parser.parse_args()
    run_batch_prediction(args.input_file, args.output_file)