import pandas as pd

def summary_by_product(seg: pd.DataFrame) -> pd.DataFrame:
    cols = ["scenario_name","product","balance","expected_loss","expected_default_balance","expected_defaults_count"]
    out = (seg[cols]
        .groupby(["scenario_name","product"], as_index=False)
        .sum(numeric_only=True))
    out["loss_rate"] = out["expected_loss"] / out["balance"]
    return out.sort_values(["scenario_name","expected_loss"], ascending=[True, False])

def summary_total(seg: pd.DataFrame) -> pd.DataFrame:
    cols = ["scenario_name","balance","expected_loss","expected_default_balance","expected_defaults_count"]
    out = (seg[cols]
        .groupby(["scenario_name"], as_index=False)
        .sum(numeric_only=True))
    out["loss_rate"] = out["expected_loss"] / out["balance"]
    return out

def monthly_view(alloc: pd.DataFrame) -> pd.DataFrame:
    cols = ["scenario_name","months_since_origination","product","expected_loss_month","expected_defaults_count_month"]
    out = (alloc[cols]
        .groupby(["scenario_name","months_since_origination","product"], as_index=False)
        .sum(numeric_only=True))
    return out.sort_values(["scenario_name","months_since_origination","product"])
