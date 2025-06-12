import pandas as pd
import joblib
from pathlib import Path
from src.preprocessing import spacy_tokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVAL_SET_PATH = PROJECT_ROOT / "data" / "evaluation_set.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "classic_model.joblib"
OUTPUT_DIR = SCRIPT_DIR / "classic_model_results"

def evaluate_classic_model():
    print("Évaluation du Modèle Classique (TF-IDF + LogReg)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Chargement du jeu d'évaluation...")
    df = pd.read_csv(EVAL_SET_PATH)
    
    print("Préparation des données...")
    text_cols = ['product_name_decli', 'brand', 'summary', 'description']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
    df[text_cols] = df[text_cols].fillna('')
    df['text_feature'] = (df['product_name_decli'] + ' ' + df['brand'] + ' ' + df['summary'] + ' ' + df['description']).str.strip()
    X_eval = df['text_feature']
    y_true = df['category']
    print("Chargement du modèle...")
    model_pipeline = joblib.load(MODEL_PATH)
    
    print("Génération des prédictions...")
    y_pred = model_pipeline.predict(X_eval)
    
    print("Génération du rapport de classification...")
    report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"Rapport sauvegardé dans {report_path}")
    
    print("Génération de la matrice de confusion...")
    labels = sorted(list(y_true.unique()))
    fig, ax = plt.subplots(figsize=(40, 40))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation='vertical', labels=labels)
    plt.title("Matrice de Confusion - Modèle Classique")
    plot_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Matrice de confusion sauvegardée dans {plot_path}")
    print("--- Évaluation terminée ---")

if __name__ == "__main__":
    evaluate_classic_model()