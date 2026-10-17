import numpy as np
import pandas as pd
from fraud_detection.features.preprocessing import TARGET, Preprocessor, n_decimals


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    v1 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionDT": np.arange(n) * 3600,
            TARGET: rng.integers(0, 2, n),
            "TransactionAmt": rng.choice([10.5, 20.25, 30.125], n),
            "ProductCD": pd.Categorical(rng.choice(["W", "C"], n)),
            "V1": v1,
            "V2": v1 * 2,  # parfaitement corrélée à V1 : doit être écartée
            "V3": rng.normal(size=n),
            "id_07": np.nan,
        }
    )


def test_n_decimals():
    s = pd.Series([10.5, 20.25, 30.125, 7.0])
    assert n_decimals(s).tolist() == [1, 2, 3, 1]


def test_no_target_or_leaky_columns():
    X = Preprocessor().fit_transform(make_df())
    for col in [TARGET, "TransactionID", "TransactionDT", "id_07"]:
        assert col not in X.columns


def test_redundant_v_removed():
    prep = Preprocessor().fit(make_df())
    assert "V1" in prep.v_kept_ and "V3" in prep.v_kept_
    assert "V2" not in prep.v_kept_


def test_unseen_category_becomes_nan():
    train = make_df()
    test = make_df(seed=1)
    test["ProductCD"] = pd.Categorical(["H"] * len(test))
    X_test = Preprocessor().fit(train).transform(test)
    assert X_test["ProductCD"].isna().all()


def test_same_columns_across_splits():
    prep = Preprocessor().fit(make_df())
    assert list(prep.transform(make_df(seed=1)).columns) == list(
        prep.transform(make_df()).columns
    )
