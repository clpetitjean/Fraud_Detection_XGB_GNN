import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"


def reduce_memory(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Réduire au maximum les colonnes numériques vers les plus petits types possibles."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")
    return df


def load_train(
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    transaction = pd.read_csv(raw_dir / "train_transaction.csv")
    identity = pd.read_csv(raw_dir / "train_identity.csv")
    df = transaction.merge(identity, on="TransactionID", how="left")
    return reduce_memory(df)


def save_interim(
    df: pd.DataFrame,
    name: str = "train_merged.parquet",
) -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / name
    df.to_parquet(path, index=False)
    return path
