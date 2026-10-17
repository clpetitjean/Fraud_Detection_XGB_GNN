"""Preprocessing des données IEEE-CIS pour XGBoost.

A partir des observations qu'on a pu faire au sein du Notebook 01_EDA, on réalise un preprocessing
de la base de données. Ce preprocessing est susceptible de changer à mesure d'études réalisées sur
la base de données original.
"""

import re

import numpy as np
import pandas as pd

TARGET = "isFraud"
SECONDS_PER_DAY = 86_400

# Colonnes jamais utilisées comme features
NON_FEATURES = ["TransactionID", "TransactionDT", TARGET, "split", "day"]

# Colonnes id_ renseignées sur moins de 1 % des transactions, sans signal propre (EDA, partie 3)
DROPPED_ID_COLS = [
    "id_07",
    "id_08",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
]


def n_decimals(amount: pd.Series) -> pd.Series:
    """Nombre de décimales du montant (1, 2 ou 3), informatif d'après l'EDA."""
    values = np.select(
        [(amount * 10).round(6) % 1 == 0, (amount * 100).round(6) % 1 == 0],
        [1, 2],
        default=3,
    )
    return pd.Series(values, index=amount.index, dtype="int8")


def select_uncorrelated_v(
    df: pd.DataFrame,
    threshold: float = 0.9,
    sample_size: int = 50_000,
    random_state: int = 0,
) -> list[str]:
    """Sélectionne les colonnes V non redondantes.

    Les colonnes V sont regroupées par nombre de valeurs manquantes (blocs identifiés
    pendant l'EDA). Dans chaque bloc, une colonne n'est conservée que si sa corrélation
    absolue avec les colonnes déjà retenues reste sous le seuil.
    """
    v_cols = [c for c in df.columns if re.fullmatch(r"V\d+", c)]
    sample = df[v_cols].sample(min(sample_size, len(df)), random_state=random_state)
    na_counts = df[v_cols].isna().sum()

    kept = []
    for _, block in na_counts.groupby(na_counts):
        cols = list(block.index)
        corr = sample[cols].corr().abs()
        block_kept = []
        for col in cols:
            if all(not (corr.loc[col, k] >= threshold) for k in block_kept):
                block_kept.append(col)
        kept.extend(block_kept)
    return kept


class Preprocessor:
    """Transforme le DataFrame brut fusionné en matrice de features pour XGBoost."""

    def __init__(self, v_corr_threshold: float = 0.9):
        self.v_corr_threshold = v_corr_threshold

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        v_cols = [c for c in df.columns if re.fullmatch(r"V\d+", c)]
        self.v_kept_ = select_uncorrelated_v(df, self.v_corr_threshold)

        excluded = set(NON_FEATURES) | set(DROPPED_ID_COLS) | set(v_cols)
        base_cols = [c for c in df.columns if c not in excluded]
        self.feature_cols_ = base_cols + self.v_kept_

        # Catégories observées à l'entraînement : toute catégorie inconnue deviendra NaN
        self.categories_ = {
            c: pd.Index(df[c].dropna().unique())
            for c in self.feature_cols_
            if isinstance(df[c].dtype, pd.CategoricalDtype)
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols_].copy()
        for col, cats in self.categories_.items():
            X[col] = pd.Categorical(X[col], categories=cats)

        X["n_decimals"] = n_decimals(df["TransactionAmt"])
        X["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
        X["weekday"] = ((df["TransactionDT"] // SECONDS_PER_DAY) % 7).astype("int8")
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
