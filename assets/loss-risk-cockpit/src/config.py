from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    outputs: Path

def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    return Paths(
        root=root,
        data=root / "data",
        outputs=root / "outputs",
    )
