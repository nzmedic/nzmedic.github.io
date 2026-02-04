#  generate customers/loans/performance with scenario levers

import math
import numpy as np
import pandas as pd
from typing import Tuple

from .config import Scenario

def sigmoid(x):
    """Compute a sigmoid transformation.

    Args:
        x: Input value or array.

    Returns:
        Sigmoid-transformed value.
    """
    return 1.0 / (1.0 + np.exp(-x))

def generate_synthetic_portfolio(
    scenario: Scenario,
    n_customers: int = 12000,
    loans_per_customer_mean: float = 1.05,
    months_max: int = 60,
    seed: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic customers, loans, and performance history.

    Args:
        scenario: Scenario configuration to apply.
        n_customers: Number of customers to simulate.
        loans_per_customer_mean: Average loans per customer (Poisson mean).
        months_max: Maximum months to simulate for each loan.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (customers_df, loans_df, perf_df). The performance DataFrame has
        one row per loan-month until graduation or scheduled close.
    """
    rng = np.random.default_rng(seed + abs(hash(scenario.name)) % 10_000)

    # Customers
    customer_id = np.arange(1, n_customers + 1)
    income = rng.lognormal(mean=np.log(70_000), sigma=0.45, size=n_customers)
    income = np.clip(income, 25_000, 250_000)

    income_stability = np.clip(rng.normal(0.6, 0.18, n_customers), 0.05, 0.98)
    tenure_months = np.clip(rng.gamma(shape=3.2, scale=18, size=n_customers), 3, 240)

    introducers = np.array(["Dealer_A", "Dealer_B", "Broker_X", "Broker_Y", "Online"])
    introducer = rng.choice(introducers, size=n_customers, p=[0.25, 0.20, 0.20, 0.20, 0.15])

    base_score = np.clip(rng.normal(590, 55, n_customers), 450, 780)

    customers_df = pd.DataFrame({
        "customer_id": customer_id,
        "income": income,
        "income_stability": income_stability,
        "tenure_months": tenure_months,
        "introducer": introducer,
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
        intro = cust.loc[cid, "introducer"]
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

        for age in range(1, term + 1):
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

            extra_pay = rng.normal(0.0, 0.015) + 0.02 * (credit_score > 690) + 0.01*(stab > 0.7)
            extra_pay = max(0.0, extra_pay)
            scheduled_paydown = balance / max(6, (term - age + 1)) * 0.18
            paydown = scheduled_paydown * (1 + extra_pay) * (1 - 0.15*dpd30)
            paydown = min(balance, max(0.0, paydown))
            balance = float(np.clip(balance - paydown, 0.0, bal0))

            prime_eligible = int((credit_score >= 680) and (ever_30dpd == 0) and (age >= 6))
            rate_diff = (apr - market_rate)

            base_hazard = sigmoid(
                -5.8
                + 1.6*prime_eligible
                + 1.2*max(0.0, rate_diff*10)
                + 1.3*(1 - balance/(bal0 + 1e-6))
                + 0.55*((credit_score - 650)/60)
                + 0.35*introducer_refi_bias.get(intro, 0.0)
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
                + 0.35*introducer_offer_bias.get(intro, 0.0)
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

            perf_rows.append({
                "scenario_name": scenario.name,
                "loan_id": int(r["loan_id"]),
                "customer_id": cid,
                "product": "vehicle",
                "origination_month": orig_m,
                "loan_age_month": age,
                "month_asof": month_asof,
                "term_months": term,
                "balance": balance,
                "apr": apr,
                "market_rate": market_rate,
                "rate_diff": rate_diff,
                "credit_score": credit_score,
                "prime_eligible": prime_eligible,
                "dpd30": dpd30,
                "late_count_roll": late_count_rolling,
                "income": inc,
                "income_stability": stab,
                "tenure_months": ten,
                "introducer": intro,
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

    perf_df = pd.DataFrame(perf_rows)
    return customers_df, loans_df, perf_df
