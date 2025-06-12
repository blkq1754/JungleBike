import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier parquet, gère les valeurs manquantes et crée une feature de texte unifiée.
    """
    if not path.exists():
        raise FileNotFoundError(f"Le fichier spécifié n'a pas été trouvé : {path}")

    df = pd.read_parquet(path)
    
    # Remplacer les NaN par une chaîne vide
    text_cols = ['product_name_decli', 'summary', 'description','brand']
    for col in text_cols:
        if col not in df.columns:
            df[col] = "" 

    df[text_cols] = df[text_cols].fillna('')

    # Créer une feature de texte unique
    df['text_feature'] = (
        df['product_name_decli'] + ' ' +
        df['summary'] + ' ' +
        df['description'] + ' ' +
        df['brand']
    )
    
    df['text_feature'] = df['text_feature'].str.strip().str.replace(r'\s+', ' ', regex=True)

    return df



class ProductDataset(Dataset):
    """
    Classe Dataset pour PyTorch. Elle prend en charge la tokenization du texte et la conversion des labels en nombres.
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_token_len: int = 128):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_token_len = max_token_len
        
        #mapping des catégories vers des ID 
        self.target_map = {label: i for i, label in enumerate(dataframe['category'].unique())}
        # décodage 
        self.inverse_target_map = {v: k for k, v in self.target_map.items()}

    def __len__(self):
        """Retourne la taille totale du dataset."""
        return len(self.dataframe)

    def __getitem__(self, index: int):
        """
        tokenization d'un élément avec son index
        """
        data_row = self.dataframe.iloc[index]
        
        text = data_row['text_feature']
        label_text = data_row['category']

        # Tokenization 
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Ajoute '[CLS]' et '[SEP]'
            max_length=self.max_token_len,
            padding='max_length',     # pour atteindre la max_length
            truncation=True,          # si le txt est trop long
            return_attention_mask=True,
            return_tensors='pt',      
        )
        
        #texte en ID
        label_id = self.target_map[label_text]

        return dict(
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.tensor(label_id, dtype=torch.long)
        )