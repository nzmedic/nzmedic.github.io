from pathlib import Path
import pandas as pd

def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file from disk.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame loaded from the CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def ensure_dir(path: Path) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path to create if missing.
    """
    path.mkdir(parents=True, exist_ok=True)
