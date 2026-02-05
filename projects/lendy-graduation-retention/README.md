# Lendy – Graduation Risk & Retention Uplift (Vehicles v1)

1. Synthetic data generation with biased observational treatment
2. Discrete-time survival modelling of loan graduation (early payoff / refinance)
3. Causal uplift modelling to estimate incremental retention from proactive rate offers
4. Explainability (global drivers + per-loan reason codes)
5. Cockpit-ready CSV outputs designed for a static GitHub Pages UI

The use case focuses on **retaining high-quality vehicle-loan customers** who are likely to “graduate” to prime lenders, while avoiding unnecessary discounts.

---

## Folder structure

| Stage    | Purpose                        | Intended Contents                 |
| -------- | ------------------------------ | --------------------------------- |
| raw      | What the source system gave us | messy, inconsistent, delayed data |
| clean    | What we trust for modelling    | imputed, standardised, validated  |
| features | Model-ready tables             | hazard_df, uplift_snapshots       |
| models   | Fitted artefacts               | pickles, coefficients             |
| eval     | Evidence                       | metrics, lift tables, calibration |




## Scenarios

Scenarios are defined in `config.py` and currently include:

- `base`
- `high_prime` (higher prime rates, weaker refi appetite)
- `low_prime` (lower prime rates, stronger refi appetite)

Each scenario is written as a separate set of CSV files and also tagged via a `scenario_name` column.

Each scenario produces a consistent set of CSV files suitable for a **scenario selector** in the Decision Cockpit UI.

| File                                     | Purpose                                                       |
| ---------------------------------------- | ------------------------------------------------------------- |
| `graduation_risk_by_loan_<scenario>.csv` | Per-loan graduation risk (3/6/12m), expected time to graduate |
| `uplift_by_loan_<scenario>.csv`          | ITE retention, incremental AUM and NII by offer size          |
| `model_metrics_<scenario>.csv`           | Survival + uplift model metrics (exec-legible)                |
| `frontier_<scenario>.csv`                | Retained AUM / NII vs budget (count + cost)                   |
| `explainability_global_<scenario>.csv`   | Global drivers of risk and uplift                             |
| `explainability_local_<scenario>.csv`    | Per-loan reason codes for cockpit drill-downs                 |




## Run

Execute the following from the repo root:

`python -m projects.lendy-graduation-retention.pipeline`

NB: outputs are pushed to `projects.lendy-graduation-retention.outputs`. To update the cockpit copy outputs to `cockpits.lendy-graduation-retention.outputs`

For quick tests (<5min) run:

Just base scenario, small dataset:
`python -m projects.lendy-graduation-retention.pipeline --scenarios base --n-customers 1500 --months-max 24`

Base + high_prime, even smaller:
`python -m projects.lendy-graduation-retention.pipeline --scenarios base,high_prime --n-customers 800 --months-max 18`

Base with some messy data
`python -m projects.lendy-graduation-retention.pipeline --scenarios base --n-customers 800 --months-max 12 --messy-level 1`

