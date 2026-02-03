#  output folder conventions + writers
# TODO: port to cockpit level for reuse and refactor other projects 

# projects/lendy-graduation-retention/io_utils.py
import os
import pandas as pd
from typing import Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def cockpit_outputs_dir() -> str:
    return os.path.join("cockpits", "lendy-graduation-retention", "outputs")

def write_outputs(
    out_dir: str,
    scenario_name: str,
    risk_by_loan: pd.DataFrame,
    uplift_by_loan: pd.DataFrame,
    model_metrics: pd.DataFrame,
    frontier: pd.DataFrame,
    explain_global: pd.DataFrame,
    explain_local: pd.DataFrame
) -> Dict[str, str]:
    ensure_dir(out_dir)

    paths = {}
    paths["graduation_risk_by_loan"] = os.path.join(out_dir, f"graduation_risk_by_loan_{scenario_name}.csv")
    paths["uplift_by_loan"] = os.path.join(out_dir, f"uplift_by_loan_{scenario_name}.csv")
    paths["model_metrics"] = os.path.join(out_dir, f"model_metrics_{scenario_name}.csv")
    paths["frontier"] = os.path.join(out_dir, f"frontier_{scenario_name}.csv")
    paths["explainability_global"] = os.path.join(out_dir, f"explainability_global_{scenario_name}.csv")
    paths["explainability_local"] = os.path.join(out_dir, f"explainability_local_{scenario_name}.csv")

    risk_by_loan.to_csv(paths["graduation_risk_by_loan"], index=False)
    uplift_by_loan.to_csv(paths["uplift_by_loan"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    frontier.to_csv(paths["frontier"], index=False)
    explain_global.to_csv(paths["explainability_global"], index=False)
    explain_local.to_csv(paths["explainability_local"], index=False)

    return paths
