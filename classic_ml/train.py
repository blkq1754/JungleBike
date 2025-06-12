# classic_ml/train.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from . import config
from . import data
from . import model

def main():
    """Fonction principale pour orchestrer l'entraînement."""
    print("--- Démarrage de l'entraînement du modèle classique ---")
    
    # Charger les données
    X, y = data.load_data(config.DATA_PATH)
    
    # Diviser les données train & test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Création du pipeline 
    pipeline = model.create_pipeline()
    
    # Entraînement
    print("Entraînement du modele")
    pipeline.fit(X_train, y_train)
    print("Entraînement terminé.")
    
    # Évaluation
    print("\n Évaluation du modele ")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # Save
    print(f"\nSauvegarde du modele dans {config.MODEL_PATH}...")
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    print("Modèle sauvegardé avec succès.")

if __name__ == "__main__":
    main()