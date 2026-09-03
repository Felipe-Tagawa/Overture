from pathlib import Path
import pandas as pd
from src.data.config import CACHE_PATH

def save_to_parquet(df: pd.DataFrame, path: Path = CACHE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Dados salvos com sucesso em :{path}")

def load_from_parquet(path: Path = CACHE_PATH) -> pd.DataFrame:
    print(f"Carregando dados do cache: {path}")
    return pd.read_parquet(path)

def cache_exists(path: Path = CACHE_PATH) -> bool:
    return path.exists()

