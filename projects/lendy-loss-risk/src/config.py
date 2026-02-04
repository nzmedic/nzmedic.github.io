from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    """Filesystem paths for the loss-risk project."""
    root: Path
    data: Path
    outputs: Path

def get_paths() -> Paths:
    """Resolve project root/data/output paths."""
    root = Path(__file__).resolve().parents[1]
    return Paths(
        root=root,
        data=root / "data",
        outputs=root / "outputs",
    )
