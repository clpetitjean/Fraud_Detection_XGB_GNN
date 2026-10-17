import pandas as pd

SECONDS_PER_DAY = 86_400


def assign_split(
    df: pd.DataFrame,
    val_days: int = 30,
    test_days: int = 30,
    time_col: str = "TransactionDT",
) -> pd.Series:
    """Attribue à chaque transaction 'train', 'val' ou 'test' selon sa date.

    Les derniers `test_days` jours forment le test, les `val_days` jours
    précédents la validation, et tout ce qui précède l'entraînement.
    """
    day = df[time_col] // SECONDS_PER_DAY
    test_start = day.max() - test_days + 1
    val_start = test_start - val_days

    split = pd.Series("train", index=df.index, name="split")
    split[day >= val_start] = "val"
    split[day >= test_start] = "test"
    return split
