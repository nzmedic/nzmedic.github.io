# writes policy outcomes based on 2 narratable policy evaluation frameworks: risk and uplift. 
# Risk evaluation focuses on model performance in predicting the outcome of interest (graduation) while uplift evaluation focuses on estimating the causal effect of an intervention (e.g., support program) on the outcome.
# This provides evidence the models are actionable.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyEvalConfig:
    """
    Configuration for simple policy evaluation artefacts.

    Args:
        top_k: Number of loans to target for Top-Uplift policy.
        risk_decile_cutoff: Risk decile threshold for risk-filtered policy (1=highest risk).
        risk_filtered_top_k: Number of loans to target within the high-risk group.
        default_bps: Discount (basis points) applied for policy valuation if not present in the input.
        horizon_months: Horizon in months for NII scaling (years = horizon/12).
    """
    top_k: int = 500
    risk_decile_cutoff: int = 1
    risk_filtered_top_k: int = 250
    default_bps: int = 100
    horizon_months: int = 12


def _value_rows(
    df: pd.DataFrame,
    bps: int,
    horizon_months: int,
) -> pd.DataFrame:
    """
    Compute incremental value columns for a chosen set of loans.

    Assumes df contains:
        balance, apr, market_rate, ite_retention
    Produces:
        ite_retention_12m, incremental_retained_balance, incremental_nii

    Args:
        df: Loan-level table with required columns.
        bps: Discount in basis points.
        horizon_months: Horizon in months for NII scaling.

    Returns:
        Copy of df with value columns added.
    """
    out = df.copy()

    scale = bps / 100.0
    out["treatment_bps"] = int(bps)

    # Match cockpit convention: clip then scale
    out["ite_retention_12m"] = out["ite_retention"].clip(-0.25, 0.40) * scale
    out["ite_retention_12m"] = out["ite_retention_12m"].clip(-0.25, 0.40)

    out["incremental_retained_balance"] = out["balance"] * out["ite_retention_12m"].clip(0, 1)

    funding_proxy = (out["market_rate"] - 0.02).clip(0.01, 0.12)
    net_margin = (out["apr"] - funding_proxy).clip(0.01, 0.25)
    discount = bps / 10_000.0
    horizon_years = horizon_months / 12.0

    out["incremental_nii"] = (
        out["incremental_retained_balance"]
        * (net_margin - discount).clip(-0.10, 0.30)
        * horizon_years
    )
    return out


def policy_outcomes_table(
    *,
    scenario_name: str,
    uplift_scored: pd.DataFrame,
    risk_by_loan: pd.DataFrame,
    config: Optional[PolicyEvalConfig] = None,
) -> pd.DataFrame:
    """
    Evaluate two simple, narratable policies and return a single outcomes table.

    Policies:
    1) Top-Uplift (Top-K by ITE retention)
    2) Risk-filtered Uplift (high-risk decile then Top-K by ITE)

    Args:
        scenario_name: Scenario identifier.
        uplift_scored: Loan-level uplift scored table (must include ite_retention, balance, apr, market_rate).
        risk_by_loan: Loan-level risk table (must include prob_graduate_12m or similar, plus loan_id).
        config: Optional policy configuration.

    Returns:
        DataFrame with columns:
            scenario_name, policy_name, n_targeted, avg_discount_bps,
            incremental_retained_balance, incremental_nii, roi, notes
    """
    cfg = config or PolicyEvalConfig()

    # Ensure required columns exist
    required = ["loan_id", "ite_retention", "balance", "apr", "market_rate"]
    missing = [c for c in required if c not in uplift_scored.columns]
    if missing:
        raise ValueError(f"uplift_scored missing required columns for policy eval: {missing}")

    # Risk score column (we use 12m for simplicity)
    risk_col = "prob_graduate_12m"
    if risk_col not in risk_by_loan.columns:
        # try a couple of fallbacks
        for alt in ["prob_graduate_6m", "prob_graduate_3m"]:
            if alt in risk_by_loan.columns:
                risk_col = alt
                break
        else:
            raise ValueError("risk_by_loan missing prob_graduate_12m (or fallback 6m/3m).")

    base = uplift_scored.copy()
    base = base.merge(risk_by_loan[["loan_id", risk_col]], on="loan_id", how="left")
    base = base.rename(columns={risk_col: "risk_prob"})

    # Drop rows that cannot be valued
    base = base.dropna(subset=["ite_retention", "balance", "apr", "market_rate"]).copy()

    rows: List[Dict] = []

    # -----------------------
    # Policy 1: Top-Uplift (Top-K by ITE)
    # -----------------------
    p1 = base.sort_values("ite_retention", ascending=False).head(cfg.top_k)
    p1v = _value_rows(p1, bps=cfg.default_bps, horizon_months=cfg.horizon_months)

    inc_bal_1 = float(p1v["incremental_retained_balance"].sum())
    inc_nii_1 = float(p1v["incremental_nii"].sum())
    roi_1 = (inc_nii_1 / max(1.0, cfg.top_k))  # simple per-target ROI proxy (kept lightweight)

    rows.append({
        "scenario_name": scenario_name,
        "policy_name": f"TopUplift_Top{cfg.top_k}",
        "n_targeted": int(len(p1v)),
        "avg_discount_bps": float(cfg.default_bps),
        "incremental_retained_balance": inc_bal_1,
        "incremental_nii": inc_nii_1,
        "roi": float(roi_1),
        "notes": "Offer to top-K loans ranked by estimated ITE retention.",
    })

    # -----------------------
    # Policy 2: Risk-filtered uplift
    #   - take highest-risk decile
    #   - within that, take Top-K by ITE
    # -----------------------
    tmp = base.sort_values("risk_prob", ascending=False).reset_index(drop=True)
    tmp["risk_decile"] = pd.qcut(tmp.index + 1, q=10, labels=False) + 1  # 1 = highest risk
    high_risk = tmp[tmp["risk_decile"] <= cfg.risk_decile_cutoff].copy()

    p2 = high_risk.sort_values("ite_retention", ascending=False).head(cfg.risk_filtered_top_k)
    p2v = _value_rows(p2, bps=cfg.default_bps, horizon_months=cfg.horizon_months)

    inc_bal_2 = float(p2v["incremental_retained_balance"].sum())
    inc_nii_2 = float(p2v["incremental_nii"].sum())
    roi_2 = (inc_nii_2 / max(1.0, len(p2v)))

    rows.append({
        "scenario_name": scenario_name,
        "policy_name": f"RiskFilteredUplift_D{cfg.risk_decile_cutoff}_Top{cfg.risk_filtered_top_k}",
        "n_targeted": int(len(p2v)),
        "avg_discount_bps": float(cfg.default_bps),
        "incremental_retained_balance": inc_bal_2,
        "incremental_nii": inc_nii_2,
        "roi": float(roi_2),
        "notes": "Offer only to high-risk loans (top decile) then rank by uplift.",
    })

    return pd.DataFrame(rows)
