import pandas as pd

def expand_timing_to_months(timing_curve: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    """Expand bucketed timing curve into monthly shares.

    Args:
        timing_curve: Timing curve DataFrame with bucket ranges.
        horizon_months: Maximum months to expand.

    Returns:
        DataFrame with month-level share_of_defaults_month.
    """
    rows = []
    for _, r in timing_curve.iterrows():
        m1, m2, share = int(r["month_start"]), int(r["month_end"]), float(r["share_of_defaults"])
        months = [m for m in range(m1, m2 + 1) if m <= horizon_months]
        if not months:
            continue
        per_month = share / len(months)
        for m in months:
            rows.append({"months_since_origination": m, "share_of_defaults_month": per_month})
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Timing curve produced empty month expansion. Check horizon_months and timing data.")
    out = out.groupby("months_since_origination", as_index=False)["share_of_defaults_month"].sum()

    # Renormalise if horizon truncates the tail
    total = out["share_of_defaults_month"].sum()
    out["share_of_defaults_month"] = out["share_of_defaults_month"] / total
    return out
