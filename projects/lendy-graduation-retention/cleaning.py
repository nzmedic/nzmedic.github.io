#Clean RAW tables and produce an issues log. 
# The cleaning rules are based on common data quality issues and general best practice. 
# Actual approach in production would depend on the specific context and requirements, and may involve more complex logic or domain-specific rules.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    """
    Configuration for cleaning rules.

    Args:
        income_clip_quantiles: Winsorisation quantiles for income.
        balance_clip_quantiles: Winsorisation quantiles for balance.
        score_min: Minimum plausible credit score.
        score_max: Maximum plausible credit score.
    """
    income_clip_quantiles: Tuple[float, float] = (0.01, 0.99)
    balance_clip_quantiles: Tuple[float, float] = (0.01, 0.99)
    score_min: float = 300.0
    score_max: float = 900.0


def _norm_str(x: object) -> str:
    """
    Normalise a string for categorical standardisation.

    Args:
        x: Any value (string-like preferred).

    Returns:
        Normalised string (lowercase, trimmed, separators unified).
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def standardise_introducer(
    series: pd.Series,
    *,
    canonical_values: Optional[List[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Standardise introducer values to a canonical set.

    Args:
        series: Raw introducer series (may be messy).
        canonical_values: Optional list of canonical values. Defaults to known set.

    Returns:
        (canonical_series, issues_df)
    """
    canon = canonical_values or ["Dealer_A", "Dealer_B", "Broker_X", "Broker_Y", "Online"]
    canon_norm = {_norm_str(c): c for c in canon}

    # known aliases / observed variants -> canonical
    alias_map = {
        "dealer_a": "Dealer_A",
        "dealer": None,  # ignored generic
        "dealer_a_": "Dealer_A",
        "dealer_a__": "Dealer_A",
        "dealer_b": "Dealer_B",
        "broker_x": "Broker_X",
        "broker_y": "Broker_Y",
        "broker-y": "Broker_Y",
        "online": "Online",
        "on_line": "Online",
    }

    raw = series.astype("string")
    norm = raw.map(_norm_str)

    out = []
    unmapped = 0
    for n, r in zip(norm.tolist(), raw.tolist()):
        if n in canon_norm:
            out.append(canon_norm[n])
        elif n in alias_map and alias_map[n] is not None:
            out.append(alias_map[n])
        elif n == "":
            out.append(np.nan)
        else:
            # keep original but flag as unmapped. 
            # TODO: assumes it is better to set to NaN and impute to "Other" than to keep unmapped value as-is. This is a judgement call and could be revisited.
            out.append(np.nan)
            unmapped += 1

    canon_series = pd.Series(out, index=series.index, name=series.name).astype("string")

    issues = []
    if unmapped > 0:
        issues.append({
            "table_name": "customers/perf",
            "column_name": "introducer",
            "issue_name": "introducer_unmapped_to_canonical",
            "severity": "warn",
            "count": int(unmapped),
            "pct": float(unmapped / len(series)) if len(series) else 0.0,
            "notes": "Unmapped introducer values were set to NaN for clean canonicalisation."
        })

    return canon_series, pd.DataFrame(issues)


def winsorise_series(
    s: pd.Series,
    q_low: float,
    q_high: float,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Winsorise a numeric series by clipping to quantile bounds.

    Args:
        s: Numeric series.
        q_low: Lower quantile.
        q_high: Upper quantile.

    Returns:
        (clipped_series, info_dict with bounds)
    """
    s_num = pd.to_numeric(s, errors="coerce")
    lo = float(s_num.quantile(q_low)) if s_num.notna().any() else np.nan
    hi = float(s_num.quantile(q_high)) if s_num.notna().any() else np.nan
    clipped = s_num.clip(lo, hi)
    return clipped, {"clip_lo": lo, "clip_hi": hi}


def clean_and_log(
    *,
    customers_raw: pd.DataFrame,
    loans_raw: pd.DataFrame,
    monthly_perf_raw: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean raw synthetic tables into model input tables and produce an issues log.

    Args:
        customers_raw: Raw customers table.
        loans_raw: Raw loans table.
        monthly_perf_raw: Raw monthly performance table.
        config: Optional cleaning config.

    Returns:
        (customers_clean, loans_clean, monthly_perf_clean, issues_log)

    Notes:
        - As-of credit score logic uses forward-fill by loan_id (no future leakage).
        - Missing indicators are created before imputation.
        - Winsorisation is applied to income and balance to reduce outlier impact.
    """
    cfg = config or CleaningConfig()
    issues: List[Dict] = []

    customers = customers_raw.copy()
    loans = loans_raw.copy()
    perf = monthly_perf_raw.copy()

    # -----------------------
    # 1) Standardise categoricals
    # -----------------------
    if "introducer" in customers.columns:
        introducer_clean, issues_intro = standardise_introducer(customers["introducer"])
        customers["introducer_clean"] = introducer_clean
        if not issues_intro.empty:
            issues.extend(issues_intro.to_dict(orient="records"))

    if "introducer" in perf.columns:
        introducer_clean_perf, issues_intro_p = standardise_introducer(perf["introducer"])
        perf["introducer_clean"] = introducer_clean_perf
        if not issues_intro_p.empty:
            issues.extend(issues_intro_p.to_dict(orient="records"))

    # -----------------------
    # 2) Missing indicators (before imputation)
    # -----------------------
    if "income" in customers.columns:
        customers["income_missing_raw"] = customers["income"].isna().astype(int)
    if "credit_score" in perf.columns:
        perf["credit_score_missing_raw"] = perf["credit_score"].isna().astype(int)
    if "introducer_clean" in customers.columns:
        customers["introducer_missing_raw"] = customers["introducer_clean"].isna().astype(int)

    # -----------------------
    # 3) Impute income (customers)
    # -----------------------
    if "income" in customers.columns:
        # prefer group median by introducer_clean if available
        if "introducer_clean" in customers.columns:
            grp = customers.groupby("introducer_clean")["income"]
            med_by_intro = grp.transform("median")
            customers["income_imputed"] = customers["income"].fillna(med_by_intro)
        else:
            customers["income_imputed"] = customers["income"]

        overall_med = float(pd.to_numeric(customers["income_imputed"], errors="coerce").median())
        missing_before = int(customers["income"].isna().sum())
        customers["income_imputed"] = customers["income_imputed"].fillna(overall_med)

        if missing_before > 0:
            issues.append({
                "table_name": "customers",
                "column_name": "income",
                "issue_name": "income_missing_imputed",
                "severity": "info",
                "count": missing_before,
                "pct": float(missing_before / len(customers)) if len(customers) else 0.0,
                "notes": "Income imputed using introducer median then overall median."
            })

        # Winsorise income
        inc_clip, bounds = winsorise_series(customers["income_imputed"], *cfg.income_clip_quantiles)
        customers["income_clean"] = inc_clip
        issues.append({
            "table_name": "customers",
            "column_name": "income_clean",
            "issue_name": "income_winsorised",
            "severity": "info",
            "count": int(inc_clip.notna().sum()),
            "pct": 1.0,
            "notes": f"Winsorised income to [{bounds['clip_lo']:.2f}, {bounds['clip_hi']:.2f}]"
        })

        # Push cleaned customer income into monthly perf (models consume perf.income)
        if "customer_id" in perf.columns and "income_clean" in customers.columns:
            inc_map = customers.set_index("customer_id")["income_clean"]
            missing_before = int(pd.to_numeric(perf.get("income", np.nan), errors="coerce").isna().sum())
            perf["income"] = pd.to_numeric(perf.get("income", np.nan), errors="coerce")
            perf["income"] = perf["income"].fillna(perf["customer_id"].map(inc_map))

            missing_after = int(perf["income"].isna().sum())
            if missing_before > 0:
                issues.append({
                    "table_name": "monthly_perf",
                    "column_name": "income",
                    "issue_name": "perf_income_filled_from_customers_income_clean",
                    "severity": "info",
                    "count": missing_before,
                    "pct": float(missing_before / len(perf)) if len(perf) else 0.0,
                    "notes": f"Filled perf.income from customers.income_clean via customer_id. Remaining missing={missing_after}."
                })

    # -----------------------
    # 4) As-of credit score logic (perf): ffill by loan_id, fallback to customer base score
    # -----------------------
    if "loan_id" in perf.columns and "month_asof" in perf.columns:
        perf = perf.sort_values(["loan_id", "month_asof"]).copy()

        # Use reported credit_score if present; otherwise credit_score_true; otherwise credit_score
        score_source = None
        if "credit_score_reported" in perf.columns:
            score_source = "credit_score_reported"
        elif "credit_score_true" in perf.columns:
            score_source = "credit_score_true"
        elif "credit_score" in perf.columns:
            score_source = "credit_score"
        else:
            score_source = None

        if score_source is not None:
            perf["credit_score_asof"] = pd.to_numeric(perf[score_source], errors="coerce")

            # as-of: forward fill within loan (no leakage)
            missing_before = int(perf["credit_score_asof"].isna().sum())
            perf["credit_score_asof"] = perf.groupby("loan_id")["credit_score_asof"].ffill()

            # fallback: customer base score (no time leakage)
            if "customer_id" in perf.columns and "base_credit_score" in customers.columns:
                base_map = customers.set_index("customer_id")["base_credit_score"]
                perf["credit_score_asof"] = perf["credit_score_asof"].fillna(perf["customer_id"].map(base_map))

            # final clip
            perf["credit_score_clean"] = perf["credit_score_asof"].clip(cfg.score_min, cfg.score_max)

            missing_after = int(perf["credit_score_clean"].isna().sum())
            if missing_before > 0:
                issues.append({
                    "table_name": "monthly_perf",
                    "column_name": "credit_score_clean",
                    "issue_name": "credit_score_missing_ffill_fallback",
                    "severity": "info",
                    "count": missing_before,
                    "pct": float(missing_before / len(perf)) if len(perf) else 0.0,
                    "notes": f"As-of score from {score_source}; ffilled by loan_id; fallback to customer base score. Remaining missing={missing_after}."
                })

    # -----------------------
    # 5) Balance winsorisation + basic imputation (perf)
    # -----------------------
    if "balance" in perf.columns and "loan_id" in perf.columns:
        perf["balance_missing_raw"] = perf["balance"].isna().astype(int)

        perf["balance_num"] = pd.to_numeric(perf["balance"], errors="coerce")
        missing_before = int(perf["balance_num"].isna().sum())

        perf["balance_num"] = perf.groupby("loan_id")["balance_num"].ffill()
        perf["balance_num"] = perf.groupby("loan_id")["balance_num"].bfill()
        perf["balance_num"] = perf["balance_num"].fillna(0.0)

        bal_clip, bounds = winsorise_series(perf["balance_num"], *cfg.balance_clip_quantiles)
        perf["balance_clean"] = bal_clip

        if missing_before > 0:
            issues.append({
                "table_name": "monthly_perf",
                "column_name": "balance_clean",
                "issue_name": "balance_missing_filled_then_winsorised",
                "severity": "info",
                "count": missing_before,
                "pct": float(missing_before / len(perf)) if len(perf) else 0.0,
                "notes": f"Balance ffill/bfill by loan_id then fill 0. Winsorised to [{bounds['clip_lo']:.2f}, {bounds['clip_hi']:.2f}]"
            })

    # -----------------------
    # 6) Assemble cleaned outputs (minimal contract)
    # -----------------------
    customers_clean = customers.copy()
    loans_clean = loans.copy()
    perf_clean = perf.copy()

    # replace model-consumed columns with cleaned versions (keep raw columns too for traceability)
    if "income_clean" in customers_clean.columns:
        customers_clean["income"] = customers_clean["income_clean"]
    if "introducer_clean" in customers_clean.columns:
        customers_clean["introducer"] = customers_clean["introducer_clean"]

    if "balance_clean" in perf_clean.columns:
        perf_clean["balance"] = perf_clean["balance_clean"]
    if "credit_score_clean" in perf_clean.columns:
        perf_clean["credit_score"] = perf_clean["credit_score_clean"]
    if "introducer_clean" in perf_clean.columns:
        perf_clean["introducer"] = perf_clean["introducer_clean"]

    issues_log = pd.DataFrame(issues)
    return customers_clean, loans_clean, perf_clean, issues_log
