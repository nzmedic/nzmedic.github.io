from pathlib import Path
import pandas as pd

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
