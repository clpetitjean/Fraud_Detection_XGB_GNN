import pandas as pd
from fraud_detection.data.split import SECONDS_PER_DAY, assign_split


def make_df(n_days=100):
    return pd.DataFrame(
        {"TransactionDT": [d * SECONDS_PER_DAY for d in range(1, n_days + 1)]}
    )


def test_split_is_chronological():
    df = make_df()
    split = assign_split(df, val_days=10, test_days=10)
    t = df["TransactionDT"]
    assert t[split == "train"].max() < t[split == "val"].min()
    assert t[split == "val"].max() < t[split == "test"].min()


def test_split_sizes():
    split = assign_split(make_df(), val_days=10, test_days=10)
    assert (split == "train").sum() == 80
    assert (split == "val").sum() == 10
    assert (split == "test").sum() == 10
