# for scenarios and shared constants
from dataclasses import dataclass
from typing import List

@dataclass
class Scenario:
    name: str
    prime_rate_shift_bps: float = 0.0
    unemployment_shift: float = 0.0
    refi_appetite_shift: float = 0.0
    offer_intensity: float = 1.0
    te_multiplier: float = 1.0

SCENARIOS: List[Scenario] = [
    Scenario(name="base"),
    Scenario(name="high_prime", prime_rate_shift_bps=+150.0, unemployment_shift=+0.01, refi_appetite_shift=-0.15,
            offer_intensity=1.0, te_multiplier=0.9),
    Scenario(name="low_prime", prime_rate_shift_bps=-150.0, unemployment_shift=-0.01, refi_appetite_shift=+0.15,
            offer_intensity=1.0, te_multiplier=1.1),
]

DEFAULT_ASOF_MONTH_RISK = 18
DEFAULT_DECISION_MONTH_UPLIFT = 18
DEFAULT_UPLIFT_HORIZON_MONTHS = 12

DEFAULT_BPS_GRID = [50, 100, 150]

# Standard feature set shared across models/explainability
BASE_FEATURE_COLS = [
    "loan_age_month", "term_months",
    "balance", "credit_score", "prime_eligible",
    "dpd30_roll3", "dpd30_roll6", "late_count_roll",
    "score_trend3", "paydown_rate1", "util_proxy",
    "rate_diff_bps",
    "income", "income_stability", "tenure_months",
    "introducer"
]
