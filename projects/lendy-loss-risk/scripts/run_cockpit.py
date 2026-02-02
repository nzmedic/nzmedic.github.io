import argparse
from pathlib import Path

from src.config import get_paths
from src.io_utils import read_csv, ensure_dir
from src.validators import (
    validate_portfolio, validate_pd, validate_lgd, validate_timing, validate_scenarios
)
from src.model_pd import attach_pd
from src.model_lgd import attach_lgd
from src.model_timing import expand_timing_to_months
from src.scenario import get_scenario, apply_scenario_to_segments, apply_timing_acceleration
from src.engine import estimate_defaults_and_losses_by_segment, allocate_losses_over_time
from src.reporting import summary_by_product, summary_total, monthly_view
from src.plotter import plot_loss_by_product, plot_monthly_loss

def main():
    parser = argparse.ArgumentParser(description="Lendy sub-portfolio loss-risk cockpit (MVP)")
    parser.add_argument("--scenario", type=str, default="base", help="Scenario name from data/scenarios.csv")
    parser.add_argument("--horizon_months", type=int, default=36, help="Months to allocate timing across")
    parser.add_argument("--horizon_years", type=float, default=1.0, help="Years horizon for PD conversion (MVP linear)")
    args = parser.parse_args()

    paths = get_paths()
    ensure_dir(paths.outputs)

    # Load data
    portfolio = read_csv(paths.data / "portfolio_snapshot.csv")
    pd_table = read_csv(paths.data / "pd_table.csv")
    lgd_table = read_csv(paths.data / "lgd_table.csv")
    timing_curve = read_csv(paths.data / "timing_curve.csv")
    scenarios = read_csv(paths.data / "scenarios.csv")

    # Validate
    validate_portfolio(portfolio)
    validate_pd(pd_table)
    validate_lgd(lgd_table)
    validate_timing(timing_curve)
    validate_scenarios(scenarios)

    # Attach PD + LGD
    segments = attach_pd(portfolio, pd_table)
    segments = attach_lgd(segments, lgd_table)

    # Timing expansion
    month_timing = expand_timing_to_months(timing_curve, horizon_months=args.horizon_months)

    # Scenario
    sc = get_scenario(scenarios, args.scenario)
    segments_scn = apply_scenario_to_segments(segments, sc)
    month_timing_scn = apply_timing_acceleration(month_timing, sc, horizon_months=args.horizon_months)

    # Engine
    seg_losses = estimate_defaults_and_losses_by_segment(segments_scn, horizon_years=args.horizon_years)
    alloc = allocate_losses_over_time(seg_losses, month_timing_scn)

    # Reports
    prod = summary_by_product(seg_losses)
    total = summary_total(seg_losses)
    monthly = monthly_view(alloc)

    # Write outputs
    prod_path = paths.outputs / f"summary_by_product_{args.scenario}.csv"
    total_path = paths.outputs / f"summary_total_{args.scenario}.csv"
    monthly_path = paths.outputs / f"monthly_view_{args.scenario}.csv"
    prod.to_csv(prod_path, index=False)
    total.to_csv(total_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    # Plots
    plot_loss_by_product(prod, paths.outputs, args.scenario)
    plot_monthly_loss(monthly, paths.outputs, args.scenario)

    print("✅ Cockpit run complete")
    print(f"- {prod_path}")
    print(f"- {total_path}")
    print(f"- {monthly_path}")
    print(f"- outputs/*.png")

if __name__ == "__main__":
    main()
