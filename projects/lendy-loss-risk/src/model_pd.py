import pandas as pd

def attach_pd(portfolio: pd.DataFrame, pd_table: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.merge(pd_table, on=["product","risk_tier"], how="left")
    missing = out["annual_pd"].isna()
    if missing.any():
        keys = out.loc[missing, ["product","risk_tier"]].drop_duplicates()
        raise ValueError(f"Missing PD for segments:\n{keys.to_string(index=False)}")
    return out
