import pandas as pd

def attach_lgd(portfolio: pd.DataFrame, lgd_table: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.merge(lgd_table, on=["product"], how="left")
    missing = out["base_lgd"].isna()
    if missing.any():
        keys = out.loc[missing, ["product"]].drop_duplicates()
        raise ValueError(f"Missing LGD for products:\n{keys.to_string(index=False)}")
    return out
