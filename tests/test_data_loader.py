import pytest
import pandas as pd
from src.data_loader import load_and_prepare_data

@pytest.fixture
def create_test_parquet_file(tmp_path):

    data = {
        'product_name_decli': ['Produit A', 'Produit B'],
        'summary': ['Résumé A', None], # Test avec une valeur manquante
        'description': ['Description A', 'Description B'],
        'category': ['Cat 1', 'Cat 2'],
        'brand': ['brand 1', 'brand 2']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.parquet"
    df.to_parquet(file_path)
    return file_path


def test_load_and_prepare_data(create_test_parquet_file):
    """
    Teste la structure et le contenu du DataFrame retourné par la fonction.
    """
    df = load_and_prepare_data(create_test_parquet_file)

    # on vérifie que la sortie est bien un DataFrame
    assert isinstance(df, pd.DataFrame)

    # on vérifie que la nouvelle colonne 'text_feature' a bien été crée
    assert 'text_feature' in df.columns

    # concaténation 
    expected_text_A = "Produit A Résumé A Description A brand 1"
    assert df.loc[0, 'text_feature'] == expected_text_A

    #les valeurs NaN 
    expected_text_B = "Produit B Description B brand 2"
    assert df.loc[1, 'text_feature'] == expected_text_B