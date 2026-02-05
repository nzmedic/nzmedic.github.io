from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DQProfileConfig:
    """
    Configuration for data quality profiling.

    Args:
        quantiles: Quantiles to compute for numeric columns.
        max_unique: Max unique values to report for categorical columns (counts still computed).
    """
    quantiles: tuple = (0.01, 0.50, 0.99)
    max_unique: int = 5000

def _profile_schema_columns() -> List[str]:
    """
    Define the output schema for column-level DQ profiles.

    Returns:
        List of column names in the expected output schema.
    """
    return [
        "table_name", "column_name", "dtype",
        "row_count", "missing_count", "missing_pct", "unique_count",
        "p01", "p50", "p99", "min", "max", "mean",
    ]

def profile_table(
    df: pd.DataFrame,
    table_name: str,
    config: Optional[DQProfileConfig] = None,
) -> pd.DataFrame:
    """
    Produce a column-level data quality profile for a DataFrame.

    Args:
        df: Input table to profile.
        table_name: Name of the table (written into output for traceability).
        config: Optional profiling configuration.

    Returns:
        DataFrame with one row per column containing row counts, missingness, and basic stats.
        If the input has zero columns, returns an empty DataFrame with the expected schema.
    """
    cfg = config or DQProfileConfig()
    schema_cols = _profile_schema_columns()

    if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return pd.DataFrame(columns=schema_cols)

    n = int(len(df))

    rows: List[Dict] = []
    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = float(missing_count / n) if n > 0 else 0.0

        dtype = str(s.dtype)
        unique_count = int(s.nunique(dropna=True)) if n > 0 else 0

        row: Dict = {
            "table_name": table_name,
            "column_name": col,
            "dtype": dtype,
            "row_count": n,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count if unique_count <= cfg.max_unique else cfg.max_unique,
        }

        # Numeric summaries
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile(list(cfg.quantiles)) if n > 0 else pd.Series(index=cfg.quantiles, dtype=float)
            row.update({
                "p01": float(q.get(cfg.quantiles[0], np.nan)),
                "p50": float(q.get(cfg.quantiles[1], np.nan)),
                "p99": float(q.get(cfg.quantiles[2], np.nan)),
                "min": float(s.min(skipna=True)) if n > 0 else np.nan,
                "max": float(s.max(skipna=True)) if n > 0 else np.nan,
                "mean": float(s.mean(skipna=True)) if n > 0 else np.nan,
            })
        else:
            row.update({
                "p01": np.nan,
                "p50": np.nan,
                "p99": np.nan,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
            })

        rows.append(row)

    out = pd.DataFrame(rows)
    
    # stable ordering, but only if columns exist to sort by
    if {"table_name", "column_name"}.issubset(out.columns) and len(out) > 0:
            out = out.sort_values(["table_name", "column_name"]).reset_index(drop=True)

    return out


def profile_many(
    tables: Dict[str, pd.DataFrame],
    config: Optional[DQProfileConfig] = None,
) -> pd.DataFrame:
    """
    Profile multiple tables and return a single stacked profile output.

    Args:
        tables: Mapping of table_name -> DataFrame.
        config: Optional profiling configuration.

    Returns:
        Concatenated profile DataFrame. Returns empty DataFrame with schema if nothing profiled.
    """
    schema_cols = _profile_schema_columns()
    if not tables:
        return pd.DataFrame(columns=schema_cols)

    frames = [profile_table(df, name, config=config) for name, df in tables.items()]
    frames = [f for f in frames if f is not None and not f.empty]

    if not frames:
        return pd.DataFrame(columns=schema_cols)

    return pd.concat(frames, ignore_index=True)

def rollup_table_profile(
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a table-level data quality rollup from a column-level profile.

    Args:
        profile_df: Output of profile_many(), containing one row per column.

    Returns:
        DataFrame with one row per table summarising completeness and scale.
    """
    if profile_df.empty:
        return pd.DataFrame()

    grp = profile_df.groupby("table_name", as_index=False)

    out = grp.agg(
        row_count=("row_count", "max"),
        column_count=("column_name", "nunique"),
        columns_with_missing=("missing_count", lambda x: int((x > 0).sum())),
        total_missing_cells=("missing_count", "sum"),
        max_missing_pct=("missing_pct", "max"),
        avg_missing_pct=("missing_pct", "mean"),
        numeric_columns=("dtype", lambda x: int(sum("int" in d or "float" in d for d in x))),
        categorical_columns=("dtype", lambda x: int(sum("object" in d or "category" in d for d in x))),
    )

    out["total_cells"] = out["row_count"] * out["column_count"]
    out["overall_missing_pct"] = (
        out["total_missing_cells"] / out["total_cells"]
    ).replace([np.inf, -np.inf], 0.0)

    return out.sort_values("table_name").reset_index(drop=True)
