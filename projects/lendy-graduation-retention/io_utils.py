# output folder conventions + writers
# TODO: port to cockpit level for reuse and refactor other projects

import os
from pathlib import Path
from typing import Dict

import pandas as pd


def repo_root() -> Path:
    """
    Resolve repo root from this file location.

    File location:
        <repo_root>/projects/lendy-graduation-retention/io_utils.py

    parents:
        [0] io_utils.py
        [1] lendy-graduation-retention
        [2] projects
        [3] <repo_root>
    """
    return Path(__file__).resolve().parents[3]


def cockpit_outputs_dir() -> str:
    """Return (and create) the cockpit outputs directory path.

    Returns:
        Output directory path as a string.
    """
    out = repo_root() / "cockpits" / "lendy-graduation-retention" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


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
    """Write scenario outputs to disk and return file paths.

    Args:
        out_dir: Output directory.
        scenario_name: Scenario name used in filenames.
        risk_by_loan: Risk-by-loan DataFrame.
        uplift_by_loan: Uplift-by-loan DataFrame.
        model_metrics: Model metrics DataFrame.
        frontier: Frontier DataFrame.
        explain_global: Global explainability DataFrame.
        explain_local: Local explainability DataFrame.

    Returns:
        Dictionary mapping output identifiers to file paths.
    """
    # Ensure the output folder exists even if out_dir was passed in
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        "graduation_risk_by_loan": os.path.join(out_dir, f"graduation_risk_by_loan_{scenario_name}.csv"),
        "uplift_by_loan": os.path.join(out_dir, f"uplift_by_loan_{scenario_name}.csv"),
        "model_metrics": os.path.join(out_dir, f"model_metrics_{scenario_name}.csv"),
        "frontier": os.path.join(out_dir, f"frontier_{scenario_name}.csv"),
        "explainability_global": os.path.join(out_dir, f"explainability_global_{scenario_name}.csv"),
        "explainability_local": os.path.join(out_dir, f"explainability_local_{scenario_name}.csv"),
    }

    risk_by_loan.to_csv(paths["graduation_risk_by_loan"], index=False)
    uplift_by_loan.to_csv(paths["uplift_by_loan"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    frontier.to_csv(paths["frontier"], index=False)
    explain_global.to_csv(paths["explainability_global"], index=False)
    explain_local.to_csv(paths["explainability_local"], index=False)

    return paths
