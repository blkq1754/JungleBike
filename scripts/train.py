import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pathlib import Path
from tqdm import tqdm
import json
from src.model import ProductClassifier

CONFIG = {
    "tokenized_data_path": Path("data/tokenized_dataset1"),
    "target_map_path": Path("models/dataset_target_map.json"),
    "saved_model_path": Path("models/best_model.bin"),
    "batch_size": 16, 
    "epochs": 2,
    "learning_rate": 2e-5,
    "test_size": 0.20 # % des données de validation
}

def run_training():
    """Fonction principale d'entraînement."""  
      
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de l'appareil : {device.upper()}")

    
    print("Chargement du dataset pré-tokenisé...")
    # Charger le dataset 
    tokenized_dataset = load_from_disk(CONFIG["tokenized_data_path"])
    
    # ensembles d'entraînement et de validation
    split_dataset = tokenized_dataset.train_test_split(
        test_size=CONFIG["test_size"],
        shuffle=True,
        stratify_by_column="label" 
    )
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"Taille du set d'entraînement : {len(train_dataset)}")
    print(f"Taille du set de validation : {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    
    # mapping pour recuperer le nombre de classes
    with open(CONFIG["target_map_path"], 'r') as f:
        target_map = json.load(f)
    n_classes = len(target_map)
    print(f"Nombre de classes à prédire : {n_classes}")
    
    model = ProductClassifier(n_classes=n_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    # FINE-TUNING
    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Époque {epoch + 1}/{CONFIG['epochs']} ---")
        
        #entraînement
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Entraînement")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # validation
        model.eval()
        val_loss = 0
        val_corrects = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                val_corrects += torch.sum(preds == labels)

        #performances de l'époque
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects.double() / len(val_dataset)
        
        print(f"Perte d'entraînement: {avg_train_loss:.4f}")
        print(f"Perte de validation: {avg_val_loss:.4f} | Précision de validation: {val_accuracy:.2%}")

    # SAUVEGARDE DU MODÈLE
    print("\nEntraînement terminé. Sauvegarde du modèle...")
    CONFIG["saved_model_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CONFIG["saved_model_path"])
    print("Modèle sauvegardé avec succès.")


if __name__ == "__main__":
    run_training()