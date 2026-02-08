# orchestration

import argparse
import json
import pandas as pd
from typing import Dict, Any
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
    build_clean_tables_and_issues,
    add_time_varying_features,
    build_discrete_time_hazard_dataset,
    time_based_split,
    build_decision_dataset_for_uplift
)
from .io_utils import (
    cockpit_outputs_dir, 
    write_outputs, 
    write_stage_table, 
    write_dq_summary, 
    write_dq_rollup, 
    write_issues_log
)

from .dq import profile_many, rollup_table_profile

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

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for running the pipeline with smaller/faster testing.

    Returns:
        argparse.Namespace with parsed values.

    Arguments:
        --scenarios: Comma-separated scenario names to run (e.g. "base,high_prime").
                        Defaults to all configured scenarios.
        --n-customers: Number of customers to simulate (smaller = faster).
        --months-max: Maximum months to simulate per loan (smaller = faster).
        --seed: Random seed for reproducibility.
        --asof-month: As-of month for hazard model train/validation split boundary.
        --decision-month: Snapshot month for uplift decision dataset.
        --uplift-horizon-months: Horizon (months) for uplift outcomes.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=str, default="",
                    help="Comma-separated scenario names to run (e.g. 'base,high_prime'). Default: all.")
    p.add_argument("--n-customers", type=int, default=12_000,
                    help="Number of customers to simulate. Default: 12000.")
    p.add_argument("--months-max", type=int, default=60,
                    help="Maximum months to simulate per loan. Default: 60.")
    p.add_argument("--seed", type=int, default=7,
                    help="Random seed. Default: 7.")
    p.add_argument("--asof-month", type=int, default=DEFAULT_ASOF_MONTH_RISK,
                    help=f"As-of month for survival modelling. Default: {DEFAULT_ASOF_MONTH_RISK}.")
    p.add_argument("--decision-month", type=int, default=DEFAULT_DECISION_MONTH_UPLIFT,
                    help=f"Decision month for uplift snapshot. Default: {DEFAULT_DECISION_MONTH_UPLIFT}.")
    p.add_argument("--uplift-horizon-months", type=int, default=DEFAULT_UPLIFT_HORIZON_MONTHS,
                    help=f"Uplift horizon in months. Default: {DEFAULT_UPLIFT_HORIZON_MONTHS}.")
    p.add_argument("--messy-level", type=int, default=0,
                    help="0 clean, 1 phase-1 messiness, 2 adds missingness/outliers/noise. Default: 0.")

    return p.parse_args()

def select_scenarios(scenarios_arg: str):
    """
    Select scenarios from config based on a comma-separated CLI argument.

    Args:
        scenarios_arg: Comma-separated scenario names. If empty, returns all SCENARIOS.

    Returns:
        List[Scenario] filtered from SCENARIOS.

    Raises:
        ValueError: If any requested scenario name is not found in SCENARIOS.
    """
    if not scenarios_arg.strip():
        return SCENARIOS

    wanted = [s.strip() for s in scenarios_arg.split(",") if s.strip()]
    by_name = {s.name: s for s in SCENARIOS}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise ValueError(f"Unknown scenario(s): {missing}. Available: {sorted(by_name.keys())}")
    return [by_name[w] for w in wanted]

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

def run_one_scenario(
    scenario,
    out_dir: str,
    seed: int = 7,
    asof_month: int = DEFAULT_ASOF_MONTH_RISK,
    decision_month: int = DEFAULT_DECISION_MONTH_UPLIFT,
    uplift_horizon_months: int = DEFAULT_UPLIFT_HORIZON_MONTHS,
    n_customers: int = 12_000,
    months_max: int = 60,
    messy_level: int = 0,
) -> Dict[str, Any]:
    """Run the full modeling pipeline for a single scenario.

    Args:
        scenario: Scenario configuration to run.
        out_dir: Output directory for cockpit CSV/JSON artifacts.
        seed: Random seed for synthetic data generation.
        asof_month: As-of month for survival modeling.
        decision_month: Decision snapshot month for uplift modeling.
        uplift_horizon_months: Horizon in months for uplift outcomes.
        messy_level: Messy-data level (0/1/2).

    Returns:
        Summary dictionary with scenario name, output paths, and row counts.
    """

    # A) generate (RAW) and write stage artefacts (per scenario)
    customers_raw, loans_raw, perf_raw = generate_synthetic_portfolio(
        scenario=scenario, 
        seed=seed,
        n_customers=n_customers,
        months_max=months_max,
        messy_level=messy_level,
    )

    write_stage_table(customers_raw, "raw", "customers_raw", scenario.name)
    write_stage_table(loans_raw, "raw", "loans_raw", scenario.name)
    write_stage_table(perf_raw, "raw", "monthly_perf_raw", scenario.name)

    raw_profile = profile_many({
        "customers_raw": customers_raw,
        "loans_raw": loans_raw,
        "monthly_perf_raw": perf_raw,
    })

    write_dq_summary(raw_profile, stage="raw", scenario_name=scenario.name)
    raw_rollup = rollup_table_profile(raw_profile)
    write_dq_rollup(raw_rollup, stage="raw", scenario_name=scenario.name)

    # B) pass through clean datasts and write stage artefacts (per scenario)
    customers_clean, loans_clean, perf_clean, issues_clean = build_clean_tables_and_issues(
        customers_raw=customers_raw,
        loans_raw=loans_raw,
        monthly_perf_raw=perf_raw,
    )

    issues_clean.insert(0, "scenario_name", scenario.name)
    write_issues_log(issues_clean, stage="clean", scenario_name=scenario.name)

    write_stage_table(customers_clean, "clean", "customers_clean", scenario.name)
    write_stage_table(loans_clean, "clean", "loans_clean", scenario.name)
    write_stage_table(perf_clean, "clean", "monthly_perf_clean", scenario.name)

    clean_profile = profile_many({
        "customers_clean": customers_clean,
        "loans_clean": loans_clean,
        "monthly_perf_clean": perf_clean,
    })

    write_dq_summary(clean_profile, stage="clean", scenario_name=scenario.name)
    clean_rollup = rollup_table_profile(clean_profile)
    write_dq_rollup(clean_rollup, stage="clean", scenario_name=scenario.name)

    # C) prep features 
    # Use cleaned perf df for everything downstream to mirror real-world contract where cleaning is a separate stage with its own outputs and quality checks.
    perf_feat = add_time_varying_features(perf_clean)
    hazard_df = build_discrete_time_hazard_dataset(perf_feat)

    train_h, valid_h = time_based_split(hazard_df, split_month=asof_month)

    #catch invalid splits that can occur with small datasets or aggressive month settings, with guidance for resolution
    if len(train_h) == 0:
        raise ValueError(
            f"Hazard training set is empty (scenario={scenario.name}). "
            f"Try reducing --asof-month (currently {asof_month}) or increasing --months-max. "
            f"hazard_df rows={len(hazard_df)}, month_asof min={hazard_df['month_asof'].min() if len(hazard_df) else None}, "
            f"max={hazard_df['month_asof'].max() if len(hazard_df) else None}."
        )
    if len(valid_h) == 0:
        raise ValueError(
            f"Hazard validation set is empty (scenario={scenario.name}). "
            f"Try increasing --asof-month or increasing --months-max."
        )

    # D) hazard models

    # assert features are not missing before modeling. Simple approach to identifying root cause of small batch failures in testing
    # TODO: . Implement more robust checks and handling for this. For example could check for NaNs immediately after feature engineering and before train/validation split, and could also add checks for expected column presence and data types.
    # could also add an assert here to check that the split resulted in non-empty train and validation sets.
    feature_cols = [
        "loan_age_month", "term_months",
        "balance", "credit_score", "prime_eligible",
        "dpd30_roll3", "dpd30_roll6", "late_count_roll",
        "score_trend3", "paydown_rate1", "util_proxy",
        "rate_diff_bps",
        "income", "income_stability", "tenure_months",
    ]

    bad = train_h[feature_cols].isna().mean().sort_values(ascending=False)
    bad = bad[bad > 0]
    if len(bad) > 0:
        raise ValueError(f"NaNs remain in hazard features:\n{bad}")

    hazard_models = fit_hazard_models(train_h, valid_h)
    hazard_prod = hazard_models["hazard_gbm"]

    risk_by_loan = loan_level_survival_summary(perf_feat, hazard_prod, asof_month, horizons=(3, 6, 12))

    # E) survival metrics
    surv_metrics = survival_eval_metrics(perf_feat, hazard_prod, asof_month)
    surv_metrics["scenario_name"] = scenario.name
    surv_metrics = surv_metrics[["scenario_name", "model_name", "metric_name", "metric_value", "notes"]]

    # F) uplift dataset + models
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


# TODO: Guardrails included so time-based split months remain valid for small test runs but do not currently but risk remains of invalid splits if 
# user sets asof_month or decision_month too high relative to months_max. Could add explicit checks and warnings for this in parse_args() or run_one_scenario().
# Could also consider adding a "--fast" flag that sets a consistent combination of n_customers, months_max, and split months for quick testing without needing to adjust multiple parameters.

def main():
    """
    Run configured scenarios and write cockpit outputs.

    Uses CLI arguments to optionally reduce synthetic portfolio size for faster testing.
    """
    args = parse_args()
    scenarios = select_scenarios(args.scenarios)

    # Guardrail: month_asof = origination_month (0-11) + loan_age_month (1..months_max) keep split/decision months within the simulated horizon. 
    # So typical max month_asof is ~ 11 + months_max.
    max_month_asof_expected = 11 + args.months_max

    asof_month = min(args.asof_month, max_month_asof_expected - 1)
    decision_month = min(args.decision_month, max_month_asof_expected - 1)

    if asof_month < 3:
        asof_month = 3
    if decision_month < 3:
        decision_month = 3


    out_dir = cockpit_outputs_dir()
    print(f"Writing cockpit outputs to: {out_dir}")

    summaries = [
        run_one_scenario(
            sc,
            out_dir=out_dir,
            seed=args.seed,
            asof_month=args.asof_month,
            decision_month=args.decision_month,
            uplift_horizon_months=args.uplift_horizon_months,
            n_customers=args.n_customers,
            months_max=args.months_max,
            messy_level=args.messy_level,
        )
        for sc in scenarios
    ]
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
