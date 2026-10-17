import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"


# Colonnes dont la précision décimale porte de l'information
KEEP_FLOAT64 = {"TransactionAmt"}


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes numériques vers des types plus légers.

    TransactionAmt reste en float64 : le nombre de décimales du montant
    est informatif (voir notebooks/01_eda.ipynb) et serait altéré en float32.
    """
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        if col not in KEEP_FLOAT64:
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


def load_interim(
    name: str = "train_merged.parquet",
) -> pd.DataFrame:
    return pd.read_parquet(INTERIM_DIR / name)
