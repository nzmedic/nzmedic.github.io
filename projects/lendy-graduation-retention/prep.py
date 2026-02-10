#feature engineering + hazard dataset + uplift decision dataset + time splits

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

from .cleaning import clean_and_log, CleaningConfig

def build_clean_tables_and_issues(
    *,
    customers_raw: pd.DataFrame,
    loans_raw: pd.DataFrame,
    monthly_perf_raw: pd.DataFrame,
    config: CleaningConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build cleaned tables from raw inputs and return an issues log.

    Args:
        customers_raw: Raw customers table.
        loans_raw: Raw loans table.
        monthly_perf_raw: Raw monthly performance table.
        config: Optional cleaning configuration.

    Returns:
        (customers_clean, loans_clean, monthly_perf_clean, issues_log)
    """
    return clean_and_log(
        customers_raw=customers_raw,
        loans_raw=loans_raw,
        monthly_perf_raw=monthly_perf_raw,
        config=config,
    )


def build_clean_tables(
    *,
    customers_raw: pd.DataFrame,
    loans_raw: pd.DataFrame,
    monthly_perf_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper returning only cleaned tables.

    Args:
        customers_raw: Raw customers table.
        loans_raw: Raw loans table.
        monthly_perf_raw: Raw monthly performance table.

    Returns:
        (customers_clean, loans_clean, monthly_perf_clean)
    """
    customers_clean, loans_clean, perf_clean, _issues = build_clean_tables_and_issues(
        customers_raw=customers_raw,
        loans_raw=loans_raw,
        monthly_perf_raw=monthly_perf_raw,
    )
    return customers_clean, loans_clean, perf_clean


def add_time_varying_features(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling, lagged, and derived features to the performance dataset.

    Args:
        perf_df: Monthly loan performance records.

    Returns:
        DataFrame with engineered features.

    Raises:
        ValueError: If required identifier columns are missing.
    """
    required = ["loan_id", "loan_age_month", "month_asof", "balance", "income", "credit_score", "apr", "market_rate", "introducer"]
    missing = [c for c in required if c not in perf_df.columns]
    if missing:
        raise ValueError(
            f"add_time_varying_features: perf_df missing required columns: {missing}. "
            f"Columns present: {list(perf_df.columns)[:30]}{'...' if len(perf_df.columns) > 30 else ''}"
        )

    df = perf_df.sort_values(["loan_id", "loan_age_month"]).copy()

    df["dpd30_roll3"] = df.groupby("loan_id")["dpd30"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["dpd30_roll6"] = df.groupby("loan_id")["dpd30"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)

    df["score_lag3"] = df.groupby("loan_id")["credit_score"].shift(3)
    df["score_trend3"] = (df["credit_score"] - df["score_lag3"]).fillna(0.0)

    df["bal_lag1"] = df.groupby("loan_id")["balance"].shift(1)
    df["bal_change1"] = (df["balance"] - df["bal_lag1"]).fillna(0.0)
    df["paydown_rate1"] = (-df["bal_change1"] / (df["bal_lag1"].replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
    df["paydown_rate1"] = df["paydown_rate1"].fillna(0.0).clip(-0.5, 1.0)

    denom = (df["income"] / 12.0).replace(0, np.nan)
    df["util_proxy"] = (df["balance"] / denom).replace([np.inf, -np.inf], np.nan)
    # Guarantee reasonable range and fill any remaining missing with median. 
    # In practice we would want to investigate and handle any extreme outliers separately, but this is a simple approach for the demo.
    df["util_proxy"] = df["util_proxy"].clip(0, 10).fillna(df["util_proxy"].median())

    df["rate_diff_bps"] = (df["apr"] - df["market_rate"]) * 10_000.0
    df["introducer"] = df["introducer"].astype("category")
    return df

def build_discrete_time_hazard_dataset(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Build a discrete-time hazard modeling dataset from performance data.

    Args:
        perf_df: Monthly loan performance records with engineered features.

    Returns:
        DataFrame with hazard model columns and an event indicator.
    """
    df = perf_df.copy()
    df["event"] = df["graduated_this_month"].astype(int)
    cols = [
        "scenario_name", "loan_id", "customer_id", "product", "month_asof", "loan_age_month", "term_months",
        "balance", "credit_score", "prime_eligible",
        "dpd30_roll3", "dpd30_roll6", "late_count_roll",
        "score_trend3", "paydown_rate1", "util_proxy",
        "rate_diff_bps", "apr", "market_rate",
        "income", "income_stability", "tenure_months", "introducer",
        "treated", "treatment_bps",
        "event"
    ]
    return df[cols].copy()

def time_based_split(df: pd.DataFrame, time_col: str = "month_asof", split_month: int = 18) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/validation sets by time.

    Args:
        df: Input dataset with a time column.
        time_col: Column to use for the split boundary.
        split_month: Inclusive month for training data.

    Returns:
        Tuple of (train_df, valid_df).
    """
    if df.empty:
        return df.copy(), df.copy()

    min_t = int(df[time_col].min())
    max_t = int(df[time_col].max())

    split_month = int(np.clip(split_month, min_t, max_t - 1)) if max_t > min_t else min_t

    train = df[df[time_col] <= split_month].copy()
    valid = df[df[time_col] > split_month].copy()

    return train, valid

def build_decision_dataset_for_uplift(perf_df: pd.DataFrame, decision_month: int, horizon_months: int = 12) -> pd.DataFrame:
    """Create a decision snapshot dataset for uplift modeling.

    Args:
        perf_df: Monthly loan performance records.
        decision_month: Snapshot month to define treatment decisions.
        horizon_months: Future months to evaluate graduation outcomes.

    Returns:
        DataFrame with decision-time features and outcome labels.
    """
    df = perf_df.sort_values(["loan_id", "month_asof"]).copy()

    snap = df[df["month_asof"] == decision_month].copy()
    snap = snap[snap["balance"] > 1_000].copy()

    future = df[(df["month_asof"] > decision_month) & (df["month_asof"] <= decision_month + horizon_months)]
    grad_within = future.groupby("loan_id")["graduated_this_month"].max().rename("graduated_within_h")
    snap = snap.merge(grad_within, on="loan_id", how="left")
    snap["graduated_within_h"] = snap["graduated_within_h"].fillna(0).astype(int)
    snap["retained_within_h"] = 1 - snap["graduated_within_h"]

    keep = [
        "scenario_name", "loan_id", "customer_id", "product", "month_asof",
        "loan_age_month", "term_months",
        "balance", "credit_score", "prime_eligible",
        "dpd30_roll3", "dpd30_roll6", "late_count_roll",
        "score_trend3", "paydown_rate1", "util_proxy",
        "rate_diff_bps", "apr", "market_rate",
        "income", "income_stability", "tenure_months", "introducer",
        "treated", "treatment_bps",
        "graduated_within_h", "retained_within_h"
    ]
    return snap[keep].copy()
