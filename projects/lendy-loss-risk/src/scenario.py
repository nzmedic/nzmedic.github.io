import pandas as pd

def get_scenario(scenarios: pd.DataFrame, scenario_name: str) -> dict:
    row = scenarios.loc[scenarios["scenario_name"] == scenario_name]
    if row.empty:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {scenarios['scenario_name'].tolist()}")
    r = row.iloc[0].to_dict()
    return {
        "scenario_name": r["scenario_name"],
        "pd_multiplier": float(r["probability_of_default_multiplier"]),
        "lgd_multiplier": float(r["loss_given_default_multiplier"]),
        "timing_acceleration": float(r["timing_acceleration"]),
    }

def apply_scenario_to_segments(segments: pd.DataFrame, sc: dict) -> pd.DataFrame:
    out = segments.copy()
    out["scenario_name"] = sc["scenario_name"]
    out["pd_annual_scn"] = (out["annual_probability_of_default"] * sc["probability_of_default_multiplier"]).clip(0, 1)
    out["lgd_scn"] = (out["base_loss_given_default"] * sc["loss_given_default_multiplier"]).clip(0, 1)
    return out

def apply_timing_acceleration(month_timing: pd.DataFrame, sc: dict, horizon_months: int) -> pd.DataFrame:
    """
    A simple acceleration: move probability mass earlier by scaling month index,
    then re-binning to integer months.
    """
    out = month_timing.copy()
    out["adj_month"] = (out["months_since_origination"] / sc["timing_acceleration"]).round().astype(int)
    out["adj_month"] = out["adj_month"].clip(lower=1, upper=horizon_months)
    out = out.groupby("adj_month", as_index=False)["share_of_defaults_month"].sum()
    out = out.rename(columns={"adj_month":"months_since_origination"})
    # renormalise
    out["share_of_defaults_month"] = out["share_of_defaults_month"] / out["share_of_defaults_month"].sum()
    return out
