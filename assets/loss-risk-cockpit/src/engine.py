import pandas as pd
import numpy as np

def estimate_defaults_and_losses_by_segment(segments: pd.DataFrame, horizon_years: float = 1.0) -> pd.DataFrame:
    """
    Convert annual PD to horizon PD (simple linear approx for MVP).
    For a more realistic approach later: convert using survival / hazard.
    """
    out = segments.copy()
    out["pd_horizon"] = (out["pd_annual_scn"] * horizon_years).clip(0, 1)
    out["expected_default_balance"] = out["balance"] * out["pd_horizon"]
    out["expected_loss"] = out["expected_default_balance"] * out["lgd_scn"]
    out["expected_defaults_count"] = (out["expected_default_balance"] / out["avg_loan_size"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return out

def allocate_losses_over_time(segments_losses: pd.DataFrame, month_timing: pd.DataFrame) -> pd.DataFrame:
    """
    Allocates expected losses and default counts over months using timing shares.
    """
    base = segments_losses.copy()
    timing = month_timing.copy()

    # Cross join segments x months
    base["key"] = 1
    timing["key"] = 1
    alloc = base.merge(timing, on="key").drop(columns=["key"])

    alloc["expected_loss_month"] = alloc["expected_loss"] * alloc["share_of_defaults_month"]
    alloc["expected_defaults_count_month"] = alloc["expected_defaults_count"] * alloc["share_of_defaults_month"]
    return alloc
