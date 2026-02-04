import pandas as pd

def attach_pd(portfolio: pd.DataFrame, pd_table: pd.DataFrame) -> pd.DataFrame:
    """Attach probability-of-default values to a portfolio snapshot.

    Args:
        portfolio: Portfolio snapshot DataFrame.
        pd_table: PD lookup table by product and risk tier.

    Returns:
        Portfolio DataFrame enriched with annual_probability_of_default.
    """
    out = portfolio.merge(pd_table, on=["product","risk_tier"], how="left")
    missing = out["annual_probability_of_default"].isna()
    if missing.any():
        keys = out.loc[missing, ["product","risk_tier"]].drop_duplicates()
        raise ValueError(f"Missing PD for segments:\n{keys.to_string(index=False)}")
    return out
