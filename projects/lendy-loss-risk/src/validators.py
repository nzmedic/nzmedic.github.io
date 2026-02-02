import pandas as pd
import numpy as np

# data tests. Would typically be part of ingestion logic (ELT) but this keeps the use case focused on the cockpit not data engineering  

def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def validate_portfolio(df: pd.DataFrame) -> None:
    require_columns(df, ["as_of_date","product","risk_tier","balance","avg_loan_size","vintage_month"], "portfolio_snapshot")
    if (df["balance"] < 0).any():
        raise ValueError("portfolio_snapshot: negative balances found")
    if (df["avg_loan_size"] <= 0).any():
        raise ValueError("portfolio_snapshot: non-positive avg_loan_size found")

def validate_pd(df: pd.DataFrame) -> None:
    require_columns(df, ["product","risk_tier","annual_pd"], "pd_table")
    if (df["annual_pd"] < 0).any() or (df["annual_pd"] > 1).any():
        raise ValueError("pd_table: annual_pd must be in [0,1]")

def validate_lgd(df: pd.DataFrame) -> None:
    require_columns(df, ["product","base_lgd"], "lgd_table")
    if (df["base_lgd"] < 0).any() or (df["base_lgd"] > 1).any():
        raise ValueError("lgd_table: base_lgd must be in [0,1]")

def validate_timing(df: pd.DataFrame) -> None:
    require_columns(df, ["bucket","month_start","month_end","share_of_defaults"], "timing_curve")
    if (df["month_start"] <= 0).any() or (df["month_end"] <= 0).any():
        raise ValueError("timing_curve: months must be positive")
    if (df["month_end"] < df["month_start"]).any():
        raise ValueError("timing_curve: month_end must be >= month_start")
    s = float(df["share_of_defaults"].sum())
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"timing_curve: shares must sum to 1.0, got {s}")

def validate_scenarios(df: pd.DataFrame) -> None:
    require_columns(df, ["scenario_name","pd_multiplier","lgd_multiplier","timing_acceleration"], "scenarios")
    if (df["pd_multiplier"] <= 0).any():
        raise ValueError("scenarios: pd_multiplier must be > 0")
    if (df["lgd_multiplier"] <= 0).any():
        raise ValueError("scenarios: lgd_multiplier must be > 0")
    if (df["timing_acceleration"] <= 0).any():
        raise ValueError("scenarios: timing_acceleration must be > 0")
