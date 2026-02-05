# for scenarios and shared constants
from dataclasses import dataclass
from typing import List
from pathlib import Path

# Classes and functions for configuration and shared constants across the project.

@dataclass(frozen=True)
class Paths:
    """
    File system paths used by the project.

    Attributes:
        project_root: Root directory of this project module.
        outputs_root: Root directory where pipeline artefacts are written.
    """
    project_root: Path
    outputs_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.outputs_root / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.outputs_root / "clean"

    @property
    def features_dir(self) -> Path:
        return self.outputs_root / "features"

    @property
    def models_dir(self) -> Path:
        return self.outputs_root / "models"

    @property
    def eval_dir(self) -> Path:
        return self.outputs_root / "eval"

    @property
    def cockpit_dir(self) -> Path:
        return self.outputs_root / "cockpit"

@dataclass(frozen=True)
class Scenario:
    """Scenario configuration for synthetic data and model adjustments.

    Attributes:
        name: Scenario identifier used in outputs.
        prime_rate_shift_bps: Shift in prime rate in basis points.
        unemployment_shift: Additive unemployment shift.
        refi_appetite_shift: Shift in refinance appetite factor.
        offer_intensity: Multiplier applied to offer probability.
        te_multiplier: Multiplier applied to treatment effects.
    """
    name: str
    prime_rate_shift_bps: float = 0.0
    unemployment_shift: float = 0.0
    refi_appetite_shift: float = 0.0
    offer_intensity: float = 1.0
    te_multiplier: float = 1.0

# Utility function to get paths based on project structure.

def get_paths() -> Paths:
    """
    Construct project path configuration.

    Returns:
        Paths instance containing project and outputs roots.
    """
    project_root = Path(__file__).resolve().parents[0]
    outputs_root = project_root / "outputs"
    return Paths(project_root=project_root, outputs_root=outputs_root)


SCENARIOS: List[Scenario] = [
    Scenario(name="base"),
    Scenario(
        name="high_prime",
        prime_rate_shift_bps=+150.0,
        unemployment_shift=+0.01,
        refi_appetite_shift=-0.15,
        offer_intensity=1.0,
        te_multiplier=0.9,
    ),
    Scenario(
        name="low_prime",
        prime_rate_shift_bps=-150.0,
        unemployment_shift=-0.01,
        refi_appetite_shift=+0.15,
        offer_intensity=1.0,
        te_multiplier=1.1,
    ),
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

