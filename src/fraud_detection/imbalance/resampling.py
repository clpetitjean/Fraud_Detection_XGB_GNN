"""Préparation des données et suréchantillonnage SMOTE.

SMOTE exige des valeurs numériques complètes. Les manquants sont imputés par la
médiane de l'entraînement, avec des indicateurs de présence pour conserver
l'information portée par leur absence (voir notebooks/01_eda.ipynb).
Les catégories sont encodées en entiers et traitées par SMOTENC.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.neighbors import NearestNeighbors


class SmotePreparer:
    """Rend la matrice de features compatible avec SMOTE, et inversement pour XGBoost."""

    def fit(self, X: pd.DataFrame) -> "SmotePreparer":
        self.cat_cols_ = [
            c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
        ]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        self.medians_ = X[self.num_cols_].median()
        self.categories_ = {c: X[c].cat.categories for c in self.cat_cols_}

        # Un seul indicateur par motif de manquants distinct
        na = X.isna()
        na = na.loc[:, na.any()]
        seen, self.indicator_cols_ = set(), []
        for col in na.columns:
            signature = np.packbits(na[col].to_numpy()).tobytes()
            if signature not in seen:
                seen.add(signature)
                self.indicator_cols_.append(col)
        return self

    @property
    def discrete_cols_(self) -> list[str]:
        """Colonnes à ne pas interpoler : indicateurs de présence et catégories."""
        return [f"{c}_na" for c in self.indicator_cols_] + self.cat_cols_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.num_cols_].fillna(self.medians_).astype(np.float32)
        ind = X[self.indicator_cols_].isna().astype(np.int16).add_suffix("_na")
        # Code 0 = manquant, catégories décalées de 1
        cat = pd.DataFrame(
            {c: X[c].cat.codes.astype(np.int16) + 1 for c in self.cat_cols_},
            index=X.index,
        )
        return pd.concat([num, ind, cat], axis=1)

    def to_xgb(self, Z: pd.DataFrame) -> pd.DataFrame:
        """Recrée les colonnes catégorielles attendues par XGBoost."""
        Z = Z.copy()
        for c in self.cat_cols_:
            codes = Z[c].round().astype(int) - 1  # -1 redevient une valeur manquante
            Z[c] = pd.Categorical.from_codes(codes, categories=self.categories_[c])
        return Z


def apply_smote(
    Z: pd.DataFrame,
    y: pd.Series,
    discrete_cols: list[str],
    method: str = "smotenc",
    sampling_strategy: float = 0.25,
    k_neighbors: int = 5,
    random_state: int = 0,
):
    """Suréchantillonne les fraudes de l'entraînement.

    `sampling_strategy` est le rapport fraudes / transactions légitimes visé.
    `method="smotenc"` gère les colonnes discrètes par vote majoritaire ;
    `method="smote"` les interpole puis les arrondit (plus rapide, moins rigoureux).
    """
    if method == "smotenc":
        mask = [c in set(discrete_cols) for c in Z.columns]
        sampler = SMOTENC(
            categorical_features=mask,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
    elif method == "smote":
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    Z_res, y_res = sampler.fit_resample(Z, y)
    Z_res[discrete_cols] = Z_res[discrete_cols].round().astype(np.int16)
    return Z_res, y_res


def _generate_synthetic(
    frauds: pd.DataFrame,
    n_new: int,
    discrete_cols: set,
    k_neighbors: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Génère `n_new` fraudes synthétiques par interpolation entre fraudes voisines."""
    num_cols = [
        c
        for c in frauds.columns
        if pd.api.types.is_numeric_dtype(frauds[c]) and c not in discrete_cols
    ]

    # On décide de choisir un voisignage qui appartient toujours à une même catégorie (ex: produit)
    dist_cols = [c for c in num_cols if frauds[c].notna().all()]
    D = frauds[dist_cols].to_numpy(dtype=np.float64)
    D = (D - D.mean(axis=0)) / (D.std(axis=0) + 1e-9)
    k = min(k_neighbors, len(frauds) - 1)
    _, neighbors = NearestNeighbors(n_neighbors=k + 1).fit(D).kneighbors(D)

    base = rng.integers(0, len(frauds), n_new)
    other = neighbors[
        base, rng.integers(1, k + 1, n_new)
    ]  # la colonne 0 est le point lui-même
    lam = rng.random(n_new)

    A = frauds.iloc[base].reset_index(drop=True)
    B = frauds.iloc[other].reset_index(drop=True)

    # Par défaut, chaque variable est copiée depuis le point le plus proche sur le segment
    synth = A.copy()
    closer_to_b = lam >= 0.5
    synth.loc[closer_to_b] = B.loc[closer_to_b]

    # On interpole quand les deux valeurs existent
    for col in num_cols:
        a, b = A[col].to_numpy(float), B[col].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        values = synth[col].to_numpy(dtype=float, copy=True)
        values[both] = a[both] + lam[both] * (b[both] - a[both])

        observed = frauds[col].dropna().to_numpy(float)
        if observed.size and np.all(np.mod(observed, 1) == 0):
            values = np.round(values)  # compteurs entiers
        synth[col] = values.astype(frauds[col].dtype)
    return synth


def product_smote(
    X: pd.DataFrame,
    y: pd.Series,
    group_col: str = "ProductCD",
    discrete_cols: tuple = (),
    sampling_strategy: float = 0.25,
    k_neighbors: int = 5,
    random_state: int = 0,
):
    """SMOTE appliqué séparément au sein de chaque produit, sans imputation.

    Pour chaque produit, des fraudes synthétiques sont générées jusqu'à atteindre
    `sampling_strategy` fraudes par transaction légitime. Les valeurs manquantes
    et les catégories sont préservées (voir notebooks/01_eda.ipynb).
    """
    rng = np.random.default_rng(random_state)
    discrete = set(discrete_cols) | {
        c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
    }
    parts = []
    for idx in X.groupby(group_col, observed=True).groups.values():
        Xg, yg = X.loc[idx], y.loc[idx]
        frauds = Xg[yg == 1]
        n_new = int(sampling_strategy * (yg == 0).sum()) - len(frauds)
        if n_new > 0 and len(frauds) >= 2:
            parts.append(_generate_synthetic(frauds, n_new, discrete, k_neighbors, rng))

    synth = pd.concat(parts, ignore_index=True)
    X_res = pd.concat([X, synth], ignore_index=True)
    y_res = pd.concat(
        [y.reset_index(drop=True), pd.Series(1, index=range(len(synth)))],
        ignore_index=True,
    ).rename(y.name)
    return X_res, y_res
