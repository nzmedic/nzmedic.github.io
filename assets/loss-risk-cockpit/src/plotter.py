from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_by_product(summary_prod: pd.DataFrame, outdir: Path, scenario_name: str) -> Path:
    df = summary_prod[summary_prod["scenario_name"] == scenario_name].copy()
    df = df.sort_values("expected_loss", ascending=False)

    fig = plt.figure()
    plt.bar(df["product"], df["expected_loss"])
    plt.title(f"Expected Loss by Product ({scenario_name})")
    plt.xlabel("Product")
    plt.ylabel("Expected Loss")
    path = outdir / f"loss_by_product_{scenario_name}.png"
    fig.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path

def plot_monthly_loss(monthly: pd.DataFrame, outdir: Path, scenario_name: str) -> Path:
    df = monthly[monthly["scenario_name"] == scenario_name].copy()
    df = df.groupby("months_since_origination", as_index=False)["expected_loss_month"].sum()

    fig = plt.figure()
    plt.plot(df["months_since_origination"], df["expected_loss_month"])
    plt.title(f"Expected Loss Over Time ({scenario_name})")
    plt.xlabel("Months Since Origination")
    plt.ylabel("Expected Loss (Month)")
    path = outdir / f"loss_over_time_{scenario_name}.png"
    fig.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path
