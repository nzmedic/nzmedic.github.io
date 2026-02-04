import pandas as pd
import numpy as np

# data tests. Would typically be part of ingestion logic (ELT) but this keeps the use case focused on the cockpit not data engineering  

def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Ensure a DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        cols: Required column names.
        name: Dataset name for error messages.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def validate_portfolio(df: pd.DataFrame) -> None:
    """Validate portfolio snapshot schema and value ranges.

    Args:
        df: Portfolio snapshot DataFrame.
    """
    require_columns(df, ["as_of_date","product","risk_tier","balance","average_loan_size","vintage_month"], "portfolio_snapshot")
    if (df["balance"] < 0).any():
        raise ValueError("portfolio_snapshot: negative balances found")
    if (df["average_loan_size"] <= 0).any():
        raise ValueError("portfolio_snapshot: non-positive average_loan_size found")

def validate_pd(df: pd.DataFrame) -> None:
    """Validate probability-of-default table.

    Args:
        df: PD lookup DataFrame.
    """
    require_columns(df, ["product","risk_tier","annual_probability_of_default"], "pd_table")
    if (df["annual_probability_of_default"] < 0).any() or (df["annual_probability_of_default"] > 1).any():
        raise ValueError("pd_table: annual_probability_of_default must be in [0,1]")

def validate_lgd(df: pd.DataFrame) -> None:
    """Validate loss-given-default table.

    Args:
        df: LGD lookup DataFrame.
    """
    require_columns(df, ["product","base_loss_given_default"], "lgd_table")
    if (df["base_loss_given_default"] < 0).any() or (df["base_loss_given_default"] > 1).any():
        raise ValueError("lgd_table: base_loss_given_default must be in [0,1]")

def validate_timing(df: pd.DataFrame) -> None:
    """Validate timing curve data.

    Args:
        df: Timing curve DataFrame.
    """
    require_columns(df, ["bucket","month_start","month_end","share_of_defaults"], "timing_curve")
    if (df["month_start"] <= 0).any() or (df["month_end"] <= 0).any():
        raise ValueError("timing_curve: months must be positive")
    if (df["month_end"] < df["month_start"]).any():
        raise ValueError("timing_curve: month_end must be >= month_start")
    s = float(df["share_of_defaults"].sum())
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"timing_curve: shares must sum to 1.0, got {s}")

def validate_scenarios(df: pd.DataFrame) -> None:
    """Validate scenario multipliers and timing acceleration values.

    Args:
        df: Scenario configuration DataFrame.
    """
    require_columns(df, ["scenario_name","probability_of_default_multiplier","loss_given_default_multiplier","timing_acceleration"], "scenarios")
    if (df["probability_of_default_multiplier"] <= 0).any():
        raise ValueError("scenarios: probability_of_default_multiplier must be > 0")
    if (df["loss_given_default_multiplier"] <= 0).any():
        raise ValueError("scenarios: loss_given_default_multiplier must be > 0")
    if (df["timing_acceleration"] <= 0).any():
        raise ValueError("scenarios: timing_acceleration must be > 0")
