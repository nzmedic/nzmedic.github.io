# Contains utilities for computing and formatting uplift evaluation metrics i.e. AUUC.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UpliftEvalConfig:
    """
    Configuration for uplift evaluation artefacts.

    Args:
        n_bins: Number of bins/segments used when producing curve tables (if added later).
    """
    n_bins: int = 10


def qini_auuc_table(
    *,
    scenario_name: str,
    uplift_scored: pd.DataFrame,
    model_name: str = "uplift_dr",
    auuc_value: Optional[float] = None,
    qini_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create a compact Qini/AUUC metrics table suitable for cockpit consumption.

    Args:
        scenario_name: Scenario identifier.
        uplift_scored: Loan-level uplift-scored dataset (used for row counts / sanity checks).
        model_name: Model identifier.
        auuc_value: Precomputed AUUC-like metric (recommended to pass from compute_uplift_curve()).
        qini_value: Optional Qini coefficient if you compute it (can be added later).

    Returns:
        DataFrame with columns:
            scenario_name, model_name, metric_name, metric_value, baseline, n_rows, notes
    """
    n_rows = int(len(uplift_scored))

    rows = []

    if auuc_value is not None:
        rows.append({
            "scenario_name": scenario_name,
            "model_name": model_name,
            "metric_name": "AUUC_like",
            "metric_value": float(auuc_value),
            "baseline": "random",
            "n_rows": n_rows,
            "notes": "Area under uplift curve vs random ordering (approx).",
        })

    if qini_value is not None:
        rows.append({
            "scenario_name": scenario_name,
            "model_name": model_name,
            "metric_name": "Qini",
            "metric_value": float(qini_value),
            "baseline": "random",
            "n_rows": n_rows,
            "notes": "Qini coefficient (optional; add when computed).",
        })

    return pd.DataFrame(rows)

def treatment_effect_comparison_table(
    *,
    scenario_name: str,
    naive_te: float,
    adjusted_te: float,
    model_name: str = "uplift_dr",
) -> pd.DataFrame:
    """
    Create a naive vs adjusted treatment effect comparison artefact.

    Args:
        scenario_name: Scenario identifier.
        naive_te: Naive treated-minus-control estimate (observational, biased).
        adjusted_te: Adjusted doubly-robust mean ITE estimate.
        model_name: Uplift model identifier.

    Returns:
        DataFrame with columns:
            scenario_name, model_name, estimate_type, mean_retention_uplift, notes
    """
    return pd.DataFrame([
        {
            "scenario_name": scenario_name,
            "model_name": model_name,
            "estimate_type": "naive",
            "mean_retention_uplift": float(naive_te),
            "notes": "Treated-minus-control retention difference (biased in observational data).",
        },
        {
            "scenario_name": scenario_name,
            "model_name": model_name,
            "estimate_type": "adjusted_dr",
            "mean_retention_uplift": float(adjusted_te),
            "notes": "Mean ITE from doubly-robust estimator (adjusted for confounding).",
        },
        {
            "scenario_name": scenario_name,
            "model_name": model_name,
            "estimate_type": "bias_gap",
            "mean_retention_uplift": float(naive_te - adjusted_te),
            "notes": "Naive minus adjusted. Positive suggests upward bias from selection.",
        },
    ])
