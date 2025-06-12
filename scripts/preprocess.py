import pandas as pd
from pathlib import Path
from datasets import Dataset, Features, ClassLabel, Value
from transformers import CamembertTokenizer
import json
from tqdm import tqdm

from src.data_loader import load_and_prepare_data

CONFIG = {
    "input_data_path": Path("data/dataset.parquet"),
    "output_data_dir": Path("data/tokenized_dataset1"),
    "target_map_path": Path("models/dataset_target_map.json"),
    "model_name": "camembert-base",
    "max_len": 128
}

def preprocess():
    print("--- Démarrage du pré-traitement ---")

    print(f"Chargement des données depuis {CONFIG['input_data_path']}...")
    df = load_and_prepare_data(CONFIG['input_data_path'])

    print("Création du mapping des catégories...")
    categories = sorted(df['category'].unique())
    target_map = {label: i for i, label in enumerate(categories)}
    
    CONFIG['target_map_path'].parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG['target_map_path'], 'w', encoding='utf-8') as f:
        json.dump(target_map, f, ensure_ascii=False, indent=4)
    print(f"Mapping sauvegardé dans {CONFIG['target_map_path']}")
    
    df['label'] = df['category'].map(target_map)
    #df = df.reset_index(drop=True) 
    #print("Colonnes du DataFrame:", df.columns.tolist())

   
    category_names = [k for k, v in sorted(target_map.items(), key=lambda item: item[1])]
    
    features = Features({
        'product_name_decli': Value('string'),
        'brand': Value('string'),
        'category': Value('string'),
        'summary': Value('string'),
        'description': Value('string'),
        'text_feature': Value('string'),
        #on spécifie le type spécial pour notre colonne 'label'
        'label': ClassLabel(names=category_names)
    })

    hg_dataset = Dataset.from_pandas(df, features=features)

    print("Chargement du tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained(CONFIG['model_name'])

    def tokenize_function(examples):
        return tokenizer(
            examples["text_feature"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_len"]
        )

    print("Tokenization du dataset en cours...")
    tokenized_dataset = hg_dataset.map(tokenize_function, batched=True, num_proc=4)
    
    columns_to_keep = ['input_ids', 'attention_mask', 'label']
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
    )
    tokenized_dataset.set_format("torch")

    print(f"Sauvegarde du dataset tokenizé dans {CONFIG['output_data_dir']}...")
    CONFIG["output_data_dir"].parent.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(CONFIG['output_data_dir'])

    print("--- Pré-traitement terminé avec succès ! ---")

if __name__ == "__main__":
    preprocess()