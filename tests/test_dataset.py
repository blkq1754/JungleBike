import pytest
import pandas as pd
import torch
from transformers import CamembertTokenizer

from src.data_loader import ProductDataset, load_and_prepare_data

@pytest.fixture
def create_test_parquet_file(tmp_path):
    """créer un fichier de données de test temporaire."""
    data = {
        'product_name_decli': ['produit a', 'Produit B', 'Produit C'],
        'summary': ['r A', None, 'r C'],
        'description': ['description A', 'description B', ''],
        'brand': ['brand 1', 'brand 2', ''],
        'category': ['cat 1', 'cat 2', 'cat 1'] 
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.parquet"
    df.to_parquet(file_path)
    return file_path

def test_product_dataset(create_test_parquet_file):
    """
    Teste l'initialisation et le fonctionnement de la classe ProductDataset.
    """
    # On charge le tokenizer correspondant à notre futur modèle
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    # On charge les données via notre fonction déjà testée
    df = load_and_prepare_data(create_test_parquet_file)    
    max_len = 32

    #l'instance de la classe à tester
    dataset = ProductDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_token_len=max_len
    )

    # La longueur du dataset doit correspondre au nombre de lignes du dataframe
    assert len(dataset) == 3

    item = dataset[0]

    # un dictionnaire
    assert isinstance(item, dict)
    
    # les clés attendues
    expected_keys = ['input_ids', 'attention_mask', 'labels']
    assert all(key in item for key in expected_keys)

    # des tenseurs PyTorch
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['attention_mask'], torch.Tensor)
    assert isinstance(item['labels'], torch.Tensor)

    #  la bonne taille 
    assert item['input_ids'].shape == (max_len,)
    assert item['attention_mask'].shape == (max_len,)

    # le label 'Cat 1' est la première catégorie => son ID doit être 0.
    assert item['labels'].item() == 0

    # 'Cat 2 => ID = 1.
    item2 = dataset[1]
    assert item2['labels'].item() == 1