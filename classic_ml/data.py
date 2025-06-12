import pandas as pd

def load_data(path):
    """Charge les données et crée la feature textuelle unifiée."""
    print("Chargement et préparation des données...")
    df = pd.read_parquet(path)
    
    text_cols = ['product_name_decli', 'summary', 'description','brand']
    for col in text_cols:
        if col not in df.columns:
            df[col] = "" # être sûr que les colonnes existent

    df[text_cols] = df[text_cols].fillna('')
    
    df['text_feature'] = (
        df['product_name_decli'] + ' ' +
        df['summary'] + ' ' +
        df['description'] + ' ' +
        df['brand']
    ).str.strip().str.replace(r'\s+', ' ', regex=True)
    
    print("Données chargées.")
    return df['text_feature'], df['category']