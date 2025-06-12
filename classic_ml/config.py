from pathlib import Path

DATA_PATH = Path("data/dataset.parquet")
MODEL_PATH = Path("models/LR_model.joblib")
TOP_LEVEL_DIR = Path(__file__).resolve().parent.parent

#dossier racine
DATA_PATH = TOP_LEVEL_DIR / DATA_PATH
MODEL_PATH = TOP_LEVEL_DIR / MODEL_PATH