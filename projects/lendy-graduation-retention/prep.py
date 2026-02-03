#feature engineering + hazard dataset + uplift decision dataset + time splits

# projects/lendy-graduation-retention/prep.py
import pandas as pd
import numpy as np
from typing import Tuple

def add_time_varying_features(perf_df: pd.DataFrame) -> pd.DataFrame:
    df = perf_df.sort_values(["loan_id", "loan_age_month"]).copy()

    df["dpd30_roll3"] = df.groupby("loan_id")["dpd30"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["dpd30_roll6"] = df.groupby("loan_id")["dpd30"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)

    df["score_lag3"] = df.groupby("loan_id")["credit_score"].shift(3)
    df["score_trend3"] = (df["credit_score"] - df["score_lag3"]).fillna(0.0)

    df["bal_lag1"] = df.groupby("loan_id")["balance"].shift(1)
    df["bal_change1"] = (df["balance"] - df["bal_lag1"]).fillna(0.0)
    df["paydown_rate1"] = (-df["bal_change1"] / (df["bal_lag1"].replace(0, np.nan))).fillna(0.0)
    df["paydown_rate1"] = df["paydown_rate1"].clip(-0.5, 1.0)

    df["util_proxy"] = df["balance"] / (df["income"] / 12.0)
    df["util_proxy"] = df["util_proxy"].clip(0, 10)

    df["rate_diff_bps"] = (df["apr"] - df["market_rate"]) * 10_000.0
    df["introducer"] = df["introducer"].astype("category")
    return df

def build_discrete_time_hazard_dataset(perf_df: pd.DataFrame) -> pd.DataFrame:
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
    train = df[df[time_col] <= split_month].copy()
    valid = df[df[time_col] > split_month].copy()
    return train, valid

def build_decision_dataset_for_uplift(perf_df: pd.DataFrame, decision_month: int, horizon_months: int = 12) -> pd.DataFrame:
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
