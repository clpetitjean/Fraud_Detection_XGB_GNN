import numpy as np
import pandas as pd
from fraud_detection.imbalance.resampling import (
    SmotePreparer,
    apply_smote,
    product_smote,
)


def make_X(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[:50] = np.nan
    return pd.DataFrame(
        {
            "a": a,
            "b": np.where(np.isnan(a), np.nan, 1.0),  # même motif de manquants que "a"
            "c": rng.normal(size=n),
            "cat": pd.Categorical(
                rng.choice(["x", "y", None], n), categories=["x", "y"]
            ),
        }
    )


def test_no_nan_and_one_indicator_per_pattern():
    prep = SmotePreparer().fit(make_X())
    Z = prep.transform(make_X())
    assert not Z.isna().any().any()
    assert prep.indicator_cols_ == ["a", "cat"]  # "b" partage le motif de "a"


def test_categories_round_trip():
    X = make_X()
    prep = SmotePreparer().fit(X)
    back = prep.to_xgb(prep.transform(X))
    pd.testing.assert_series_equal(back["cat"], X["cat"], check_names=False)


def test_smote_reaches_ratio_and_keeps_discrete_valid():
    X = make_X()
    y = pd.Series([1] * 30 + [0] * 270)
    prep = SmotePreparer().fit(X)
    Z_res, y_res = apply_smote(
        prep.transform(X), y, prep.discrete_cols_, sampling_strategy=0.5
    )
    assert y_res.sum() == 135
    assert Z_res["cat"].isin([0, 1, 2]).all()
    assert Z_res["a_na"].isin([0, 1]).all()


def make_products(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    product = np.array(["W"] * 200 + ["C"] * 200)
    df = pd.DataFrame(
        {
            "ProductCD": pd.Categorical(product),
            "amt": rng.gamma(2, 50, n),
            "count": rng.integers(0, 10, n).astype(float),
            "card1": rng.choice([1000, 2000, 3000], n),
            "id_x": np.where(
                product == "W", np.nan, rng.normal(size=n)
            ),  # jamais renseignée pour W
        }
    )
    y = pd.Series([1] * 20 + [0] * 180 + [1] * 30 + [0] * 170, name="isFraud")
    return df, y


def test_ratio_per_product():
    X, y = make_products()
    X_res, y_res = product_smote(X, y, discrete_cols=("card1",), sampling_strategy=0.5)
    for p, n_legit in [("W", 180), ("C", 170)]:
        mask = X_res["ProductCD"] == p
        assert y_res[mask].sum() == int(0.5 * n_legit)


def test_structural_missing_preserved():
    X, y = make_products()
    X_res, _ = product_smote(X, y, discrete_cols=("card1",), sampling_strategy=0.5)
    assert X_res.loc[X_res["ProductCD"] == "W", "id_x"].isna().all()


def test_discrete_and_integer_columns_valid():
    X, y = make_products()
    X_res, _ = product_smote(X, y, discrete_cols=("card1",), sampling_strategy=0.5)
    assert X_res["card1"].isin([1000, 2000, 3000]).all()
    assert (X_res["count"] % 1 == 0).all()
