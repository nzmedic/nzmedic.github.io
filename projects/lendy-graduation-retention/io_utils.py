# output folder conventions + writers
from __future__ import annotations

# TODO: port to cockpit level for reuse and refactor other projects

import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# -----------------------------
# Path management
# -----------------------------


def repo_root() -> Path:
    """
    Resolve repository root by walking upward until common repo markers are found.

    Args:
        None.

    Returns:
        Path to the repository root directory.

    Raises:
        RuntimeError: If no repo root marker is found within a reasonable number of parents.
    """
    here = Path(__file__).resolve()

    for p in [here] + list(here.parents):
        if (p / "projects").is_dir() and (p / "cockpits").is_dir():
            return p
        if (p / ".gitignore").exists():
            return p

    raise RuntimeError(f"Could not locate repo root from {here}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cockpit_outputs_dir() -> str:
    """Return (and create) the cockpit outputs directory path."""
    out = repo_root() / "cockpits" / "lendy-graduation-retention" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# -----------------------------
# Stage IO (raw/clean/features + data quality summaries)
# -----------------------------

def project_outputs_root() -> Path:
    """
    Project outputs root:
        <repo_root>/projects/lendy-graduation-retention/outputs
    """
    return repo_root() / "projects" / "lendy-graduation-retention" / "outputs"


def stage_dir(stage: str) -> Path:
    """
    e.g. stage_dir("raw") -> .../outputs/raw
    """
    out = project_outputs_root() / stage
    ensure_dir(out)
    return out


def stage_table_path(stage: str, table_name: str, scenario_name: Optional[str] = None) -> Path:
    """
    Mirrors your existing cockpit naming convention:
        <table>_<scenario>.csv

    Examples:
        stage_table_path("raw", "monthly_perf_raw", "base")
        -> .../outputs/raw/monthly_perf_raw_base.csv
        stage_table_path("clean", "loans_clean", "high_prime")
        -> .../outputs/clean/loans_clean_high_prime.csv
    """
    if scenario_name:
        fname = f"{table_name}_{scenario_name}.csv"
    else:
        fname = f"{table_name}.csv"
    return stage_dir(stage) / fname


def write_stage_table(df: pd.DataFrame, stage: str, table_name: str, scenario_name: str) -> str:
    """
    Write stage table to the stage folder for easy retrieval by downstream stages and cockpit writers.
    Also helpful for debugging and manual inspection of intermediate artefacts.

    Args:
        df: Stage artefact DataFrame.
        stage: Stage name (e.g., "raw" or "clean").
        table_name: Base table name (no scenario suffix).
        scenario_name: Scenario identifier.

    Returns:
        File path written.
    """
    path = stage_table_path(stage, table_name, scenario_name)
    df.to_csv(path, index=False)
    return str(path)


def read_stage_table(stage: str, table_name: str, scenario_name: str) -> pd.DataFrame:
    path = stage_table_path(stage, table_name, scenario_name)
    return pd.read_csv(path)


def write_dq_summary(profile_df: pd.DataFrame, stage: str, scenario_name: str) -> str:
    """
    Write a DQ summary profile for a stage (raw/clean).

    Args:
        profile_df: Output of dq.profile_many/profile_table.
        stage: Stage name (e.g., "raw" or "clean").
        scenario_name: Scenario identifier used in the filename.

    Returns:
        File path written.
    """
    path = stage_table_path(stage, f"data_quality_{stage}_summary", scenario_name)
    profile_df.to_csv(path, index=False)
    return str(path)

def write_dq_rollup(rollup_df: pd.DataFrame, stage: str, scenario_name: str) -> str:
    """
    Write a table-level DQ rollup for a stage (raw/clean).

    Args:
        rollup_df: Output of dq.rollup_table_profile().
        stage: Stage name (e.g., "raw" or "clean").
        scenario_name: Scenario identifier used in the filename.

    Returns:
        File path written.
    """
    path = stage_table_path(stage, f"data_quality_{stage}_rollup", scenario_name)
    rollup_df.to_csv(path, index=False)
    return str(path)

def write_issues_log(issues_df: pd.DataFrame, stage: str, scenario_name: str) -> str:
    """
    Write a structured issues log for a pipeline stage.

    Args:
        issues_df: Issues log DataFrame (rule-based).
        stage: Stage name, typically "clean".
        scenario_name: Scenario identifier used in filename.

    Returns:
        File path written.
    """
    path = stage_table_path(stage, f"issues_log_{stage}", scenario_name)
    issues_df.to_csv(path, index=False)
    return str(path)

# model evaluation artefacts (risk eval, uplift eval, frontier eval, explainability eval)
def write_eval_table(df: pd.DataFrame, table_name: str, scenario_name: str) -> str:
    """
    Write an evaluation artefact table to the eval stage folder.
    This could be implemented within write_stage_table with stage="eval". Currently separated for clarity and the potential for divergent logic.

    Args:
        df: Evaluation artefact DataFrame.
        table_name: Base table name (no scenario suffix).
        scenario_name: Scenario identifier.

    Returns:
        File path written.
    """
    path = stage_table_path("eval", table_name, scenario_name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


# -----------------------------
# Cockpit writers
# -----------------------------

# TODO: make these more flexible and consistent with stage writers, e.g. by accepting scenario_name and using stage_table_path 
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
    """Write scenario outputs to disk and return file paths."""
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
