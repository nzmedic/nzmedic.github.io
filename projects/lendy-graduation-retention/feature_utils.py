"""Shared feature engineering helpers for the graduation-retention project."""

import pandas as pd

from .config import BASE_FEATURE_COLS


def one_hot_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode base features used across models and explainability.

    Args:
        df: Input DataFrame containing base feature columns.

    Returns:
        DataFrame with one-hot encoded categorical features.
    """
    X = df[BASE_FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=["introducer"], drop_first=True)
    return X
