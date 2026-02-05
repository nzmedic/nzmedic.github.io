#  generate customers/loans/performance with scenario levers

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

from .config import Scenario



# Define the expected schema for the performance DataFrame (one row per loan-month) so feature engineering and modeling 
# can rely on these columns existing even if the input is empty or missing some columns due to messiness.
PERF_SCHEMA_COLS = [
    "scenario_name",
    "loan_id", "customer_id", "product",
    "origination_month",
    "loan_age_month", "month_asof", "term_months",
    "balance", "balance_true", "apr", "market_rate", "rate_diff",
    "credit_score", "credit_score_true", "credit_score_reported",
    "prime_eligible",
    "dpd30", "late_count_roll",
    "income", "income_stability", "tenure_months",
    "introducer", "introducer_canonical",
    "treated", "treatment_bps",
    "true_base_hazard", "true_hazard",
    "graduated_this_month", "closed_this_month", "is_open",
    "latent_type_true",
]



# --------------
# Helpers for synthetic data generation 
# --------------

def sigmoid(x):
    """Compute a sigmoid transformation.

    Args:
        x: Input value or array.

    Returns:
        Sigmoid-transformed value.
    """
    return 1.0 / (1.0 + np.exp(-x))

# --------------
# Helpers responsible for mesiness, noise, and missingness in the synthetic data 
# --------------

def messify_category(rng: np.random.Generator, value: str, variants: Dict[str, List[str]], p: float) -> str:
    """
    Randomly replace a canonical category with messy variants.

    Args:
        rng: Numpy RNG.
        value: Canonical category value.
        variants: Mapping canonical -> list of messy strings.
        p: Probability of applying messiness.

    Returns:
        Possibly-messy category string.
    """
    if rng.random() >= p:
        return value
    opts = variants.get(value)
    return rng.choice(opts) if opts else value


def apply_score_reporting_lag(
    rng: np.random.Generator,
    true_score_series: List[float],
    lag_min: int = 1,
    lag_max: int = 3,
) -> List[float]:
    """
    Convert a true credit score time series into a reported series with a random lag.

    Lag is implemented as "you see the score from L months ago", but always defined by
    falling back to the earliest available observation.

    Args:
        rng: Numpy RNG.
        true_score_series: List of true credit score values by month (1..T).
        lag_min: Minimum lag in months.
        lag_max: Maximum lag in months.

    Returns:
        List of reported credit scores by month (same length as input).
    """
    if not true_score_series:
        return []

    lag = int(rng.integers(lag_min, lag_max + 1))
    reported = []
    for t in range(len(true_score_series)):
        src = max(0, t - lag)
        reported.append(true_score_series[src])
    return reported


def maybe_truncate_history(
    rng: np.random.Generator,
    max_age: int,
    p_truncate: float = 0.08,
    min_keep: int = 6,
) -> int:
    """
    Possibly truncate a simulated loan history to create partial histories.

    Args:
        rng: Numpy RNG.
        max_age: Maximum months the simulation would otherwise run.
        p_truncate: Probability of truncation for a given loan.
        min_keep: Minimum months to retain if truncated.

    Returns:
        The final max age to simulate (<= max_age).
    """
    max_age = int(max_age)
    if max_age <= 1:
        return 1

    min_keep = int(min(min_keep, max_age))
    if rng.random() >= p_truncate:
        return max_age

    # truncate somewhere between min_keep and max_age (inclusive)
    return int(rng.integers(min_keep, max_age + 1))


def sigmoid01(x: float) -> float:
    """
    Numerically stable-ish sigmoid to (0,1).

    Args:
        x: Input value.

    Returns:
        Sigmoid(x) in (0,1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def maybe_set_missing_mar(
    rng: np.random.Generator,
    value: float,
    p: float,
) -> float:
    """
    Return NaN with probability p (MAR mechanism determined upstream).

    Args:
        rng: Numpy RNG.
        value: Value to possibly null out.
        p: Probability of missingness.

    Returns:
        Value or np.nan.
    """
    return np.nan if rng.random() < p else value


def add_measurement_noise(
    rng: np.random.Generator,
    value: float,
    sigma: float,
    round_to: Optional[float] = None,
) -> float:
    """
    Apply additive Gaussian noise and optional rounding.

    Args:
        rng: Numpy RNG.
        value: Base value.
        sigma: Std dev of Gaussian noise.
        round_to: If provided, round to nearest this unit (e.g. 1000.0).

    Returns:
        Noisy (and possibly rounded) value.
    """
    v = float(value + rng.normal(0.0, sigma))
    if round_to and round_to > 0:
        v = round(v / round_to) * round_to
    return v


# --------------
# Primary synthetic data generation function 
# --------------

# TODO: refactor this function to:
# 1. be more modular and testable, and to allow generating partial histories with reporting lag without needing to simulate the full history first. The main barrier is that currently the true credit score series is generated on the fly within the loan-month loop, but it needs to be generated first in order to apply the reporting lag before writing rows. One option is to generate the full true score series for all loans upfront in a separate pass, then apply lags, then generate rows in a final pass. This would also allow us to add measurement noise and MAR missingness to the scores in a more controlled way.
# 2. optimize performance by vectorizing more of the operations and avoiding Python loops where possible. The current implementation is straightforward but may be slow for large n_customers and months_max. We could explore using NumPy arrays or even a library like Numba to speed up the inner loop over loan-months.
# 3. break down the function into smaller helper functions for clarity. The current implementation is a bit monolithic and could benefit from being decomposed into logical steps (e.g. generate customers, generate loans, simulate performance with features and outcomes, apply messiness).

def generate_synthetic_portfolio(
    scenario: Scenario,
    n_customers: int = 12000,
    loans_per_customer_mean: float = 1.05,
    months_max: int = 60,
    seed: int = 7,
    messy_level: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic customers, loans, and performance history.

    Args:
        scenario: Scenario configuration to apply.
        n_customers: Number of customers to simulate.
        loans_per_customer_mean: Average loans per customer (Poisson mean).
        months_max: Maximum months to simulate for each loan.
        seed: Random seed for reproducibility.
        messy_level: Controls messy-data behaviour.
            0 = clean/idealised (current behaviour).
            1 = phase-1 messiness (safe): inconsistent categoricals, delayed score, partial histories.
            2 = phase-2 messiness (adds): MAR missingness, outliers, measurement noise

    Returns:
        Tuple of (customers_df, loans_df, perf_df). The performance DataFrame has
        one row per loan-month until graduation or scheduled close.
    """
    rng = np.random.default_rng(seed + abs(hash(scenario.name)) % 10_000)

    # Customers
    customer_id = np.arange(1, n_customers + 1)
    income = rng.lognormal(mean=np.log(70_000), sigma=0.45, size=n_customers)
    income = np.clip(income, 25_000, 250_000)

    if messy_level >= 2:
        # Observed income: noisy + rounded to nearest $1,000 to mimic a common approach in CRMs
        income_observed = np.array(
            [add_measurement_noise(rng, v, sigma=1500.0, round_to=1000.0) for v in income],
            dtype=float
        )
    else:
        income_observed = income.copy()

    # MAR: income missing more often for low stability and broker-originated customers
    if messy_level >= 2:
        
        income_mar = []
        for inc_obs, stab, intro in zip(income_observed, income_stability, introducer_canonical):
            p = sigmoid01(-3.0 + 2.3*(0.55 - stab) + (0.9 if "Broker" in intro else 0.0))
            p = float(np.clip(p, 0.0, 0.30))
            income_mar.append(maybe_set_missing_mar(rng, inc_obs, p))
        income_observed = np.array(income_mar, dtype=float)

    income_stability = np.clip(rng.normal(0.6, 0.18, n_customers), 0.05, 0.98)
    tenure_months = np.clip(rng.gamma(shape=3.2, scale=18, size=n_customers), 3, 240)

    introducers = np.array(["Dealer_A", "Dealer_B", "Broker_X", "Broker_Y", "Online"])
    introducer_variants = {
    "Dealer_A": ["Dealer A", "dealer_a", "DEALER-A", "Dealer_A ", " dealer_a"],
    "Dealer_B": ["Dealer B", "dealer_b", "DEALER-B", "Dealer_B  "],
    "Broker_X": ["Broker X", "broker_x", "BROKER-X", "Broker_X"],
    "Broker_Y": ["Broker Y", "broker_y", "BROKER-Y", "Broker-Y"],
    "Online":   ["online", "ONLINE", "On-line", "Online "],
    }

    introducer_canonical = rng.choice(introducers, size=n_customers, p=[0.25, 0.20, 0.20, 0.20, 0.15])
    if messy_level >= 1:
        introducer_observed = [
            messify_category(rng, v, introducer_variants, p=0.25) for v in introducer_canonical
        ]
    else:
        introducer_observed = introducer_canonical

    base_score = np.clip(rng.normal(590, 55, n_customers), 450, 780)

    customers_df = pd.DataFrame({
        "customer_id": customer_id,
        "income": income_observed,  # observed income with messiness if enabled, otherwise same as true income
        "income": income, # true (clean) income without messiness
        "income_stability": income_stability,
        "tenure_months": tenure_months,
        "introducer": introducer_observed,
        "base_credit_score": base_score
    })

    # Loans
    n_loans = rng.poisson(lam=loans_per_customer_mean, size=n_customers) + 1
    loan_rows = []
    loan_id_counter = 1

    for cid, k in zip(customer_id, n_loans):
        for _ in range(k):
            term = int(rng.choice([24, 36, 48, 60], p=[0.15, 0.35, 0.30, 0.20]))
            orig_balance = float(np.clip(rng.lognormal(np.log(22_000), 0.5), 6_000, 85_000))
            apr = float(np.clip(rng.normal(0.135, 0.035), 0.06, 0.24))
            orig_month = int(rng.integers(0, 12))
            loan_rows.append((loan_id_counter, cid, "vehicle", term, orig_balance, apr, orig_month))
            loan_id_counter += 1

    loans_df = pd.DataFrame(loan_rows, columns=[
        "loan_id", "customer_id", "product", "term_months", "orig_balance", "apr", "origination_month"
    ])

    perf_rows = []
    base_market_rate = 0.085
    market_rate_trend = rng.normal(0.0, 0.0006)

    introducer_refi_bias = {
        "Dealer_A": -0.10, "Dealer_B": -0.05, "Broker_X": +0.10, "Broker_Y": +0.06, "Online": +0.03
    }
    introducer_offer_bias = {
        "Dealer_A": +0.08, "Dealer_B": +0.05, "Broker_X": +0.02, "Broker_Y": +0.00, "Online": -0.03
    }

    latent_types = np.array(["sure_stay", "sure_leave", "persuadable", "do_not_disturb"])
    cust = customers_df.set_index("customer_id")

    for _, r in loans_df.iterrows():
        cid = int(r["customer_id"])
        intro_canon = cust.loc[cid, "introducer_canonical"] if "introducer_canonical" in cust.columns else cust.loc[cid, "introducer"]
        intro_obs = cust.loc[cid, "introducer"]

        base_sc = float(cust.loc[cid, "base_credit_score"])
        stab = float(cust.loc[cid, "income_stability"])
        ten = float(cust.loc[cid, "tenure_months"])
        inc = float(cust.loc[cid, "income"])

        term = int(r["term_months"])
        bal0 = float(r["orig_balance"])
        apr = float(r["apr"])
        orig_m = int(r["origination_month"])

        z_leave = (base_sc - 600)/55 + (math.log(inc) - math.log(70_000))/0.6 + (ten - 48)/60
        z_stay  = -(base_sc - 600)/70 - (math.log(inc) - math.log(70_000))/0.8 - (ten - 48)/80
        z_pers  = +1.2*(stab - 0.6) - 0.3*((base_sc - 600)/80) + 0.2*((ten - 48)/100)
        z_dnd   = -0.8*(stab - 0.6) + 0.1*((base_sc - 600)/90)

        logits = np.array([z_stay, z_leave, z_pers, z_dnd])
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        ltype = rng.choice(latent_types, p=probs)

        balance = bal0
        credit_score = base_sc + rng.normal(0, 8)
        ever_30dpd = 0
        late_count_rolling = 0
        closed = False

        loan_month_rows = []
        true_scores = []

        # To avoid simulating excessively long histories for loans that would never graduate, we cap the maximum age at either the term or a fixed month limit.
        # This also limits the size of the generated dataset for faster iteration during development.
        max_age = min(term, months_max)
        max_age = maybe_truncate_history(rng, max_age, p_truncate=0.08, min_keep=6) if messy_level >= 1 else max_age

        for age in range(1, max_age + 1):
            
            month_asof = orig_m + age
            market_rate = (base_market_rate + scenario.prime_rate_shift_bps/10_000.0) \
                          + market_rate_trend * month_asof + rng.normal(0, 0.002)

            util = balance / (inc / 12.0)
            p_delin = sigmoid(
                -2.2
                + 2.0*(0.65 - stab)
                + 0.9*(util - 2.5)
                + 1.2*scenario.unemployment_shift*30
                + rng.normal(0, 0.25)
            )
            dpd30 = int(rng.random() < p_delin)
            ever_30dpd = max(ever_30dpd, dpd30)
            late_count_rolling = int(0.7*late_count_rolling + dpd30*3)

            score_delta = (
                + 3.0*(1 - dpd30)*(0.75 - (credit_score/900.0))
                - 12.0*dpd30
                + 1.2*(stab - 0.6)
                + rng.normal(0, 2.0)
            )
            credit_score = float(np.clip(credit_score + score_delta, 430, 820))
            true_scores.append(credit_score) # keep a record of ground truth before reporting lag

            if messy_level >= 2 and rng.random() < 0.002:
                # Rare bureau shock / correction
                credit_score = float(np.clip(credit_score + rng.normal(0, 90), 430, 820))

            extra_pay = rng.normal(0.0, 0.015) + 0.02 * (credit_score > 690) + 0.01*(stab > 0.7)
            extra_pay = max(0.0, extra_pay)
            scheduled_paydown = balance / max(6, (term - age + 1)) * 0.18
            paydown = scheduled_paydown * (1 + extra_pay) * (1 - 0.15*dpd30)
            paydown = min(balance, max(0.0, paydown))
            balance = float(np.clip(balance - paydown, 0.0, bal0))

            if messy_level >= 2 and rng.random() < 0.001:
                # Rare servicing/reversal correction causing balance jump
                balance = float(np.clip(balance * rng.uniform(0.85, 1.25), 0.0, bal0 * 1.4))

            prime_eligible = int((credit_score >= 680) and (ever_30dpd == 0) and (age >= 6))
            rate_diff = (apr - market_rate)

            base_hazard = sigmoid(
                -5.8
                + 1.6*prime_eligible
                + 1.2*max(0.0, rate_diff*10)
                + 1.3*(1 - balance/(bal0 + 1e-6))
                + 0.55*((credit_score - 650)/60)
                + 0.35*introducer_refi_bias.get(intro_canon, 0.0)
                + 0.55*scenario.refi_appetite_shift
                + rng.normal(0, 0.30)
            )

            in_offer_window = int(6 <= age <= 24 and balance > 2_000)
            offer_logit = (
                -3.1
                + 2.2*in_offer_window
                + 1.6*prime_eligible
                + 1.1*max(0.0, rate_diff*10)
                + 0.6*((credit_score - 650)/70)
                + 0.45*(balance/30_000)
                + 0.35*introducer_offer_bias.get(intro_canon, 0.0)
                + rng.normal(0, 0.35)
            )
            p_offer = sigmoid(offer_logit) * scenario.offer_intensity
            p_offer = float(np.clip(p_offer, 0.0, 0.95))
            treated = int(rng.random() < p_offer)

            treatment_bps = 0
            if treated:
                w150 = float(np.clip(base_hazard*3.0, 0.05, 0.85))
                w100 = float(np.clip(0.5 + (base_hazard-0.15), 0.10, 0.75))
                w50 = 1.0
                weights = np.array([w50, w100, w150])
                weights = weights / weights.sum()
                treatment_bps = int(rng.choice([50, 100, 150], p=weights))

            te = 0.0
            if treated and in_offer_window:
                if ltype == "persuadable":
                    te = (0.35 + 0.15*(stab - 0.6) + 0.15*prime_eligible + 0.10*max(0.0, rate_diff*8))
                    te *= (treatment_bps / 100.0)
                    te *= scenario.te_multiplier
                elif ltype == "do_not_disturb":
                    te = -0.05 * (treatment_bps/100.0)
                else:
                    te = 0.02 * (treatment_bps/100.0) if ltype == "sure_stay" else 0.01 * (treatment_bps/100.0)

            hazard = float(np.clip(base_hazard * (1 - te), 0.0001, 0.999)) if (treated and in_offer_window) else base_hazard

            grad = int((not closed) and (rng.random() < hazard) and (balance > 0.0))
            scheduled_close = int((not closed) and (balance <= 100.0))

            if messy_level >= 2:
                # Observed balance: small noise + cents stripped (ledger vs reporting mismatch)
                balance_obs = float(add_measurement_noise(rng, balance, sigma=max(5.0, 0.002 * bal0), round_to=10.0))
                balance_obs = float(np.clip(balance_obs, 0.0, bal0 * 1.5))
            else:
                balance_obs = balance

            loan_month_rows.append({
                "scenario_name": scenario.name,
                "loan_id": int(r["loan_id"]),
                "customer_id": cid,
                "product": "vehicle",
                "origination_month": orig_m,
                "loan_age_month": age,
                "month_asof": month_asof,
                "term_months": term,
                "balance": balance_obs, # observed (noisy)
                "balance_true": balance, # truth
                "apr": apr,
                "market_rate": market_rate,
                "rate_diff": rate_diff,
                "credit_score_true": credit_score,
                "credit_score": credit_score,  # default to clean, overwritten if lag (messy) enabled
                "prime_eligible": prime_eligible,
                "dpd30": dpd30,
                "late_count_roll": late_count_rolling,
                "income": inc,
                "income_stability": stab,
                "tenure_months": ten,
                "introducer": intro_obs, # observed (messy), intro_canon contains ground truth if required
                "introducer_canonical": intro_canon, # canonical (clean),
                "treated": treated,
                "treatment_bps": treatment_bps,
                "true_base_hazard": base_hazard,
                "true_hazard": hazard,
                "graduated_this_month": grad,
                "closed_this_month": int(grad or scheduled_close),
                "is_open": int(not (grad or scheduled_close)),
                "latent_type_true": ltype
            })

            if grad or scheduled_close:
                closed = True
                break

        # Apply reporting lag after we have the true series
        if messy_level >= 1 and loan_month_rows:
            reported_scores = apply_score_reporting_lag(rng, true_scores, lag_min=1, lag_max=3)
            for i, rs in enumerate(reported_scores):
                loan_month_rows[i]["credit_score"] = float(rs)           # reported used by models
                loan_month_rows[i]["credit_score_reported"] = float(rs)  # explicit
        else:
            for row in loan_month_rows:
                row["credit_score_reported"] = float(row["credit_score_true"])
        
        # Persist this loan's rows into the full performance table
        perf_rows.extend(loan_month_rows)

    perf_df = pd.DataFrame(perf_rows)

    # Ensure stable schema even if perf_rows is empty or keys were missing.
    for col in PERF_SCHEMA_COLS:
        if col not in perf_df.columns:
            perf_df[col] = np.nan
    
    perf_df = perf_df[PERF_SCHEMA_COLS].copy()

    # DQ test to catch invalid outputs early during development, particularly for small test runs.
    if perf_df.empty:
        raise ValueError(
            f"generate_synthetic_portfolio produced 0 perf rows "
            f"(scenario={scenario.name}, n_customers={n_customers}, months_max={months_max}, messy_level={messy_level}). "
            f"This indicates a generator bug (e.g., loan_month_rows not appended)."
        )

    return customers_df, loans_df, perf_df
