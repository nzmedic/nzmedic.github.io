# orchestration only

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    SCENARIOS,
    DEFAULT_ASOF_MONTH_RISK,
    DEFAULT_DECISION_MONTH_UPLIFT,
    DEFAULT_UPLIFT_HORIZON_MONTHS,
    DEFAULT_BPS_GRID
)
from .synthetic import generate_synthetic_portfolio
from .prep import (
    add_time_varying_features,
    build_discrete_time_hazard_dataset,
    time_based_split,
    build_decision_dataset_for_uplift
)
from .models import (
    fit_hazard_models,
    loan_level_survival_summary,
    survival_eval_metrics,
    fit_propensity_model,
    fit_t_learner_outcome_models,
    doubly_robust_ite_retention,
    segment_customers,
    naive_vs_adjusted_treatment_effect,
    compute_uplift_curve,
    uplift_frontier
)
from .explain import (
    explain_risk_model_global_local,
    explain_uplift_via_surrogate
)
from .io_utils import cockpit_outputs_dir, write_outputs

def build_uplift_slider_rows(uplift_scored: pd.DataFrame, bps_grid, horizon_months: int):
    """Expand uplift scores across a grid of offer basis points.

    Args:
        uplift_scored: DataFrame with per-loan uplift estimates.
        bps_grid: Iterable of basis point values to evaluate.
        horizon_months: Horizon in months for incremental value projections.

    Returns:
        DataFrame with uplift rows replicated per basis point offer.
    """
    slider_rows = []
    for bps in bps_grid:
        tmp = uplift_scored.copy()
        scale = bps / 100.0
        tmp["treatment_bps"] = bps
        tmp["ite_retention_12m"] = tmp["ite_retention"].clip(-0.25, 0.40) * scale
        tmp["ite_retention_12m"] = tmp["ite_retention_12m"].clip(-0.25, 0.40)

        tmp["incremental_retained_balance"] = tmp["balance"] * tmp["ite_retention_12m"].clip(0, 1)

        funding_proxy = (tmp["market_rate"] - 0.02).clip(0.01, 0.12)
        net_margin = (tmp["apr"] - funding_proxy).clip(0.01, 0.25)
        discount = bps / 10_000.0
        horizon_years = horizon_months / 12.0
        tmp["incremental_nii"] = tmp["incremental_retained_balance"] * (net_margin - discount).clip(-0.10, 0.30) * horizon_years

        slider_rows.append(tmp[[
            "scenario_name", "loan_id", "month_asof", "treatment_bps",
            "ite_retention_12m", "incremental_retained_balance", "incremental_nii", "segment",
            "balance", "apr", "market_rate", "rate_diff_bps",
            "propensity", "mu0_retention", "mu1_retention"
        ]])

    return pd.concat(slider_rows, ignore_index=True)

def run_one_scenario(scenario, out_dir: str, seed: int = 7,
                    asof_month: int = DEFAULT_ASOF_MONTH_RISK,
                    decision_month: int = DEFAULT_DECISION_MONTH_UPLIFT,
                    uplift_horizon_months: int = DEFAULT_UPLIFT_HORIZON_MONTHS):
    """Run the full modeling pipeline for a single scenario.

    Args:
        scenario: Scenario configuration to run.
        out_dir: Output directory for cockpit CSV/JSON artifacts.
        seed: Random seed for synthetic data generation.
        asof_month: As-of month for survival modeling.
        decision_month: Decision snapshot month for uplift modeling.
        uplift_horizon_months: Horizon in months for uplift outcomes.

    Returns:
        Summary dictionary with scenario name, output paths, and row counts.
    """

    # A) generate
    _, _, perf = generate_synthetic_portfolio(scenario=scenario, seed=seed)

    # B) prep/features
    perf_feat = add_time_varying_features(perf)
    hazard_df = build_discrete_time_hazard_dataset(perf_feat)

    train_h, valid_h = time_based_split(hazard_df, split_month=asof_month)

    # C1) hazard models
    hazard_models = fit_hazard_models(train_h, valid_h)
    hazard_prod = hazard_models["hazard_gbm"]

    risk_by_loan = loan_level_survival_summary(perf_feat, hazard_prod, asof_month, horizons=(3, 6, 12))

    # D) survival metrics
    surv_metrics = survival_eval_metrics(perf_feat, hazard_prod, asof_month)
    surv_metrics["scenario_name"] = scenario.name
    surv_metrics = surv_metrics[["scenario_name", "model_name", "metric_name", "metric_value", "notes"]]

    # C2) uplift dataset + models
    uplift_base = build_decision_dataset_for_uplift(perf_feat, decision_month, horizon_months=uplift_horizon_months)
    train_u, _ = train_test_split(uplift_base, test_size=0.35, random_state=42, stratify=uplift_base["treated"])

    prop = fit_propensity_model(train_u)
    outm = fit_t_learner_outcome_models(train_u, outcome_col="retained_within_h")

    uplift_scored = doubly_robust_ite_retention(uplift_base, prop, outm, outcome_col="retained_within_h")
    uplift_scored = segment_customers(uplift_scored)

    uplift_by_loan = build_uplift_slider_rows(uplift_scored, DEFAULT_BPS_GRID, uplift_horizon_months)

    naive, adjusted = naive_vs_adjusted_treatment_effect(uplift_scored)
    _, auuc = compute_uplift_curve(uplift_scored)

    uplift_metrics = pd.DataFrame([
        {"scenario_name": scenario.name, "model_name": "uplift_dr", "metric_name": "Naive_TreatedMinusControl_Retention",
        "metric_value": naive, "notes": "biased observational estimate"},
        {"scenario_name": scenario.name, "model_name": "uplift_dr", "metric_name": "Adjusted_DR_Mean_ITE_Retention",
        "metric_value": adjusted, "notes": "doubly robust mean ITE"},
        {"scenario_name": scenario.name, "model_name": "uplift_dr", "metric_name": "AUUC_like",
        "metric_value": auuc, "notes": "uplift over random ordering (approx)"},
    ])

    # Frontier
    frontier_count, _ = uplift_frontier(uplift_scored, budget_type="count",
                                        budget_values=[100, 250, 500, 1000, 1500, 2000],
                                        horizon_months=uplift_horizon_months)
    frontier_cost, _ = uplift_frontier(uplift_scored, budget_type="cost",
                                        budget_values=[5_000, 10_000, 25_000, 50_000, 100_000, 150_000],
                                        horizon_months=uplift_horizon_months)

    frontier = pd.concat([frontier_count, frontier_cost], ignore_index=True)
    frontier.insert(0, "scenario_name", scenario.name)

    # Explainability
    explain_df = perf_feat[(perf_feat["month_asof"] == asof_month) & (perf_feat["balance"] > 1_000)].copy()
    risk_global, risk_local = explain_risk_model_global_local(hazard_prod, explain_df, model_kind="graduation_risk", top_k=8)

    uplift_global, uplift_local = explain_uplift_via_surrogate(uplift_scored, top_k=8)

    explain_global = pd.concat([risk_global, uplift_global], ignore_index=True)
    explain_global.insert(0, "scenario_name", scenario.name)
    explain_local = pd.concat([risk_local, uplift_local], ignore_index=True)
    explain_local.insert(0, "scenario_name", scenario.name)

    model_metrics = pd.concat([surv_metrics, uplift_metrics], ignore_index=True)

    paths = write_outputs(
        out_dir=out_dir,
        scenario_name=scenario.name,
        risk_by_loan=risk_by_loan,
        uplift_by_loan=uplift_by_loan,
        model_metrics=model_metrics,
        frontier=frontier,
        explain_global=explain_global,
        explain_local=explain_local
    )

    return {"scenario": scenario.name, "paths": paths, "rows_perf": int(len(perf_feat))}

def main():
    """Run all configured scenarios and write cockpit outputs."""
    out_dir = cockpit_outputs_dir()
    print(f"Writing cockpit outputs to: {out_dir}")
    summaries = [run_one_scenario(sc, out_dir=out_dir, seed=7) for sc in SCENARIOS]
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
