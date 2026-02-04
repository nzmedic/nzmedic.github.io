import pandas as pd

def attach_lgd(portfolio: pd.DataFrame, lgd_table: pd.DataFrame) -> pd.DataFrame:
    """Attach loss-given-default values to a portfolio snapshot.

    Args:
        portfolio: Portfolio snapshot DataFrame.
        lgd_table: LGD lookup table by product.

    Returns:
        Portfolio DataFrame enriched with base_loss_given_default.
    """
    out = portfolio.merge(lgd_table, on=["product"], how="left")
    missing = out["base_loss_given_default"].isna()
    if missing.any():
        keys = out.loc[missing, ["product"]].drop_duplicates()
        raise ValueError(f"Missing LGD for products:\n{keys.to_string(index=False)}")
    return out
