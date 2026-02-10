#---------------------------------------------------------
# Description: stores functions to build risk evaluation artefacts:
# - build_observed_outcomes_by_horizon: constructs observed graduation labels by horizon from performance data
# - risk_time_split_summary: summarizes time split boundaries and event rates for train/validation sets
# - approx_concordance_index: approximates Harrell's C-index by sampling comparable pairs
# - risk_auc_by_horizon: computes AUC for predicted horizon risks
# - calibration_bins: builds calibration bins (deciles) of predicted vs observed event rates
# - decile_lift_table: builds a decile lift table with cumulative event capture
#---------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class RiskEvalConfig:
    """
    Configuration for risk evaluation artefacts.

    Args:
        horizons: Horizons (months) for binary outcome evaluation.
        n_calibration_bins: Number of bins for calibration/decile lift (default 10).
        min_balance: Minimum balance filter applied at as-of snapshot to focus on active loans.
        approx_cindex_pairs: Number of random comparable pairs used to approximate C-index.
        seed: Random seed for sampling reproducibility.
    """
    horizons: Tuple[int, ...] = (3, 6, 12)
    n_calibration_bins: int = 10
    min_balance: float = 1000.0
    approx_cindex_pairs: int = 200_000
    seed: int = 7


def _asof_snapshot(perf_feat: pd.DataFrame, asof_month: int, min_balance: float) -> pd.DataFrame:
    """
    Extract the as-of month snapshot used for horizon evaluation.

    Args:
        perf_feat: Monthly performance with engineered features.
        asof_month: Month_asof snapshot month.
        min_balance: Minimum balance filter to keep meaningful active loans.

    Returns:
        Snapshot DataFrame (one row per loan at asof_month).
    """
    snap = perf_feat[(perf_feat["month_asof"] == asof_month) & (perf_feat["balance"] > min_balance)].copy()
    # Keep one row per loan_id defensively
    snap = snap.sort_values(["loan_id", "loan_age_month"]).drop_duplicates("loan_id", keep="last")
    return snap


def build_observed_outcomes_by_horizon(
    perf_feat: pd.DataFrame,
    asof_month: int,
    horizons: Iterable[int],
    min_balance: float = 1000.0,
) -> pd.DataFrame:
    """
    Build observed graduation-within-horizon labels from future performance.

    Args:
        perf_feat: Monthly performance with engineered features.
        asof_month: Snapshot month (month_asof).
        horizons: Iterable of horizons (months) e.g. [3,6,12].
        min_balance: Balance filter at snapshot.

    Returns:
        DataFrame with columns:
            loan_id, y_3m, y_6m, y_12m, time_to_event, event_observed
        Where y_Hm is 1 if graduated within horizon months after asof_month.
    """
    snap = _asof_snapshot(perf_feat, asof_month, min_balance=min_balance)
    loan_ids = snap["loan_id"].unique()

    fut = perf_feat[perf_feat["loan_id"].isin(loan_ids) & (perf_feat["month_asof"] > asof_month)].copy()
    fut = fut.sort_values(["loan_id", "month_asof"])

    # time-to-event in months after asof (censored if never graduates in observed window)
    first_grad = (
        fut[fut["graduated_this_month"] == 1]
        .groupby("loan_id")["month_asof"]
        .min()
        .rename("grad_month")
    )

    out = pd.DataFrame({"loan_id": loan_ids})
    out = out.merge(first_grad, on="loan_id", how="left")

    out["event_observed"] = out["grad_month"].notna().astype(int)
    out["time_to_event"] = np.where(
        out["event_observed"] == 1,
        out["grad_month"].astype(float) - float(asof_month),
        np.nan,
    )

    # Binary horizon outcomes
    for h in horizons:
        col = f"y_{h}m"
        out[col] = ((out["event_observed"] == 1) & (out["time_to_event"] <= float(h))).astype(int)

    out = out.drop(columns=["grad_month"])
    return out


def risk_time_split_summary(
    train_h: pd.DataFrame,
    valid_h: pd.DataFrame,
    scenario_name: str,
    time_col: str = "month_asof",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Confirm the time split boundaries and event rates.

    Args:
        train_h: Hazard training dataset (time-split).
        valid_h: Hazard validation dataset (time-split).
        scenario_name: Scenario identifier.
        time_col: Time column name.
        event_col: Event indicator column name.

    Returns:
        Summary DataFrame with one row per split (train/validation).
    """
    def _summ(df: pd.DataFrame, split: str) -> Dict:
        if df.empty:
            return {
                "scenario_name": scenario_name,
                "split_type": split,
                "min_month_asof": np.nan,
                "max_month_asof": np.nan,
                "n_rows": 0,
                "n_events": 0,
                "event_rate": np.nan,
            }
        n = int(len(df))
        e = int(df[event_col].sum()) if event_col in df.columns else 0
        return {
            "scenario_name": scenario_name,
            "split_type": split,
            "min_month_asof": int(df[time_col].min()),
            "max_month_asof": int(df[time_col].max()),
            "n_rows": n,
            "n_events": e,
            "event_rate": float(e / n) if n else np.nan,
        }

    return pd.DataFrame([_summ(train_h, "train"), _summ(valid_h, "validation")])


def approx_concordance_index(
    event_time: np.ndarray,
    event_observed: np.ndarray,
    risk_score: np.ndarray,
    n_pairs: int = 200_000,
    seed: int = 7,
) -> float:
    """
    Approximate Harrell's C-index by sampling comparable pairs.

    This avoids O(n^2) computation and keeps runtime reasonable.

    Args:
        event_time: Array of time-to-event in months (NaN for censored).
        event_observed: 1 if event observed else 0.
        risk_score: Predicted risk score (higher = more likely to graduate sooner).
        n_pairs: Number of random pairs to sample.
        seed: RNG seed.

    Returns:
        Approximate C-index in [0,1] or NaN if insufficient comparable pairs.
    """
    rng = np.random.default_rng(seed)
    n = len(risk_score)
    if n < 2:
        return np.nan

    idx = np.arange(n)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    ti, tj = event_time[i], event_time[j]
    ei, ej = event_observed[i], event_observed[j]
    si, sj = risk_score[i], risk_score[j]

    # Comparable if at least one is an observed event and has smaller time than the other (censoring ignored for the other)
    comp_ij = (ei == 1) & np.isfinite(ti) & (np.isfinite(tj) | (ej == 0))
    comp_ji = (ej == 1) & np.isfinite(tj) & (np.isfinite(ti) | (ei == 0))

    # We only score pairs where we can order times:
    # For comp_ij, require ti < tj if tj is finite; if tj is censored (NaN), treat as ti < censored_end (count as comparable)
    valid_ij = comp_ij & ((~np.isfinite(tj)) | (ti < tj))
    valid_ji = comp_ji & ((~np.isfinite(ti)) | (tj < ti))

    # For each valid direction, concordant if higher risk for earlier event
    concord = 0.0
    ties = 0.0
    total = 0.0

    # i earlier than j
    if valid_ij.any():
        total += float(valid_ij.sum())
        concord += float((si[valid_ij] > sj[valid_ij]).sum())
        ties += float((si[valid_ij] == sj[valid_ij]).sum())

    # j earlier than i
    if valid_ji.any():
        total += float(valid_ji.sum())
        concord += float((sj[valid_ji] > si[valid_ji]).sum())
        ties += float((sj[valid_ji] == si[valid_ji]).sum())

    if total == 0:
        return np.nan
    return float((concord + 0.5 * ties) / total)

# TODO: make column expectations more flexible, e.g. by accepting a mapping of horizon -> pred_col prefix and allowing fallback to generic names if expected ones not found
def risk_auc_by_horizon(
    risk_by_loan: pd.DataFrame,
    outcomes: pd.DataFrame,
    scenario_name: str,
    horizons: Iterable[int],
    dataset_label: str = "validation",
) -> pd.DataFrame:
    """
    Compute AUC for predicted horizon risks.

    Args:
        risk_by_loan: Per-loan predictions (includes loan_id and predicted horizon probs).
        outcomes: Observed outcomes by horizon (loan_id + y_{h}m).
        scenario_name: Scenario identifier.
        horizons: Horizons to evaluate.
        dataset_label: Label for dataset (typically "validation").

    Returns:
        Long-form DataFrame with AUC per horizon.
    """
    df = risk_by_loan.merge(outcomes, on="loan_id", how="inner")

    rows = []
    for h in horizons:
        y = df[f"y_{h}m"].astype(int)
        pred_col = f"prob_graduate_{h}m"
        if pred_col not in df.columns:
            # fall back if you named them differently
            pred_col = f"risk_{h}m" if f"risk_{h}m" in df.columns else pred_col

        p = pd.to_numeric(df[pred_col], errors="coerce")

        # AUC requires both classes present
        if y.nunique() < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y, p))

        rows.append({
            "scenario_name": scenario_name,
            "model_name": "hazard_gbm",
            "dataset": dataset_label,
            "horizon_months": int(h),
            "auc": auc,
        })

    return pd.DataFrame(rows)


def calibration_bins(
    risk_by_loan: pd.DataFrame,
    outcomes: pd.DataFrame,
    scenario_name: str,
    horizons: Iterable[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build calibration bins (deciles): avg predicted vs observed event rate.

    Args:
        risk_by_loan: Per-loan predictions (loan_id + predicted horizon probs).
        outcomes: Observed outcomes by horizon (loan_id + y_{h}m).
        scenario_name: Scenario identifier.
        horizons: Horizons to evaluate.
        n_bins: Number of bins (10 for deciles).

    Returns:
        DataFrame with calibration rows per horizon and bin.
    """
    df = risk_by_loan.merge(outcomes, on="loan_id", how="inner")

    frames = []
    for h in horizons:
        pred_col = f"prob_graduate_{h}m"
        if pred_col not in df.columns:
            pred_col = f"risk_{h}m" if f"risk_{h}m" in df.columns else pred_col

        tmp = df[["loan_id", pred_col, f"y_{h}m"]].copy()
        tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")

        # If too many ties or low N, qcut can fail; fall back to cut on rank
        try:
            tmp["risk_bin"] = pd.qcut(tmp[pred_col], q=n_bins, labels=False, duplicates="drop") + 1
        except Exception:
            tmp["risk_bin"] = pd.qcut(tmp[pred_col].rank(method="average"), q=n_bins, labels=False, duplicates="drop") + 1

        g = tmp.groupby("risk_bin", as_index=False).agg(
            n_loans=("loan_id", "count"),
            avg_predicted_prob=(pred_col, "mean"),
            observed_event_rate=(f"y_{h}m", "mean"),
        )

        g.insert(0, "scenario_name", scenario_name)
        g.insert(1, "horizon_months", int(h))
        g.rename(columns={"risk_bin": "risk_decile"}, inplace=True)
        frames.append(g)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decile_lift_table(
    risk_by_loan: pd.DataFrame,
    outcomes: pd.DataFrame,
    scenario_name: str,
    horizons: Iterable[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build a decile lift table with cumulative event capture.

    Args:
        risk_by_loan: Per-loan predictions (loan_id + predicted horizon probs).
        outcomes: Observed outcomes by horizon (loan_id + y_{h}m).
        scenario_name: Scenario identifier.
        horizons: Horizons to evaluate.
        n_bins: Number of bins (10 for deciles).

    Returns:
        DataFrame with lift rows per horizon and decile.
    """
    df = risk_by_loan.merge(outcomes, on="loan_id", how="inner")

    frames = []
    for h in horizons:
        pred_col = f"prob_graduate_{h}m"
        if pred_col not in df.columns:
            pred_col = f"risk_{h}m" if f"risk_{h}m" in df.columns else pred_col

        tmp = df[["loan_id", pred_col, f"y_{h}m"]].copy()
        tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")

        # rank into deciles high->low (decile 1 = highest risk)
        tmp = tmp.sort_values(pred_col, ascending=False).reset_index(drop=True)
        tmp["risk_decile"] = pd.qcut(tmp.index + 1, q=n_bins, labels=False) + 1

        g = tmp.groupby("risk_decile", as_index=False).agg(
            n_loans=("loan_id", "count"),
            n_events=(f"y_{h}m", "sum"),
            event_rate=(f"y_{h}m", "mean"),
        )

        total_events = float(g["n_events"].sum()) if g["n_events"].sum() > 0 else np.nan
        avg_rate = float(tmp[f"y_{h}m"].mean()) if len(tmp) else np.nan

        g = g.sort_values("risk_decile").copy()
        g["cumulative_event_capture"] = (g["n_events"].cumsum() / total_events) if np.isfinite(total_events) else np.nan
        g["lift_vs_average"] = (g["event_rate"] / avg_rate) if np.isfinite(avg_rate) and avg_rate > 0 else np.nan

        g.insert(0, "scenario_name", scenario_name)
        g.insert(1, "horizon_months", int(h))
        frames.append(g)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def risk_eval_artefacts(
    *,
    scenario_name: str,
    perf_feat: pd.DataFrame,
    risk_by_loan: pd.DataFrame,
    train_h: pd.DataFrame,
    valid_h: pd.DataFrame,
    asof_month: int,
    config: Optional[RiskEvalConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Produce all Step 5A risk evaluation artefacts as DataFrames.

    Args:
        scenario_name: Scenario identifier.
        perf_feat: Monthly performance with engineered features (clean).
        risk_by_loan: Per-loan predicted graduation probabilities by horizon.
        train_h: Hazard training dataset.
        valid_h: Hazard validation dataset.
        asof_month: As-of month used for scoring.
        config: Optional evaluation config.

    Returns:
        Dict mapping artefact_name -> DataFrame.
    """
    cfg = config or RiskEvalConfig()
    outcomes = build_observed_outcomes_by_horizon(
        perf_feat=perf_feat,
        asof_month=asof_month,
        horizons=cfg.horizons,
        min_balance=cfg.min_balance,
    )

    artefacts: Dict[str, pd.DataFrame] = {}

    artefacts["risk_time_split_summary"] = risk_time_split_summary(
        train_h=train_h, valid_h=valid_h, scenario_name=scenario_name
    )

    # Approx C-index using 12m risk score (or closest horizon available)
    h_for_c = 12 if 12 in cfg.horizons else int(list(cfg.horizons)[-1])
    pred_col = f"prob_graduate_{h_for_c}m"
    if pred_col not in risk_by_loan.columns:
        pred_col = f"risk_{h_for_c}m" if f"risk_{h_for_c}m" in risk_by_loan.columns else pred_col

    cdf = risk_by_loan.merge(outcomes, on="loan_id", how="inner")
    rs = pd.to_numeric(cdf[pred_col], errors="coerce").to_numpy()
    et = cdf["time_to_event"].to_numpy()
    eo = cdf["event_observed"].astype(int).to_numpy()

    c_index = approx_concordance_index(
        event_time=et,
        event_observed=eo,
        risk_score=rs,
        n_pairs=cfg.approx_cindex_pairs,
        seed=cfg.seed,
    )
    artefacts["risk_c_index"] = pd.DataFrame([{
        "scenario_name": scenario_name,
        "model_name": "hazard_gbm",
        "dataset": "validation",
        "c_index": c_index,
        "notes": f"Approx C-index using {cfg.approx_cindex_pairs} sampled pairs and risk={pred_col}"
    }])

    artefacts["risk_auc_by_horizon"] = risk_auc_by_horizon(
        risk_by_loan=risk_by_loan,
        outcomes=outcomes,
        scenario_name=scenario_name,
        horizons=cfg.horizons,
        dataset_label="validation",
    )

    artefacts["risk_calibration"] = calibration_bins(
        risk_by_loan=risk_by_loan,
        outcomes=outcomes,
        scenario_name=scenario_name,
        horizons=cfg.horizons,
        n_bins=cfg.n_calibration_bins,
    )

    artefacts["risk_decile_lift"] = decile_lift_table(
        risk_by_loan=risk_by_loan,
        outcomes=outcomes,
        scenario_name=scenario_name,
        horizons=cfg.horizons,
        n_bins=cfg.n_calibration_bins,
    )

    return artefacts
