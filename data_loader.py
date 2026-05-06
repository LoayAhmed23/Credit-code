"""
Data loading utilities.
Reads monthly CSVs from prime_data/ and transaction XLSX files from
transaction_data/, enforces dtypes, cleans numeric strings, parses dates,
tags each row with its source month, and provides a merge helper.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Helpers for parsing messy numeric strings (e.g. "1,234.00")
# ---------------------------------------------------------------------------

def _parse_int(x):
    if pd.isna(x):
        return pd.NA
    x_clean = str(x).strip().replace(",", "")
    if x_clean.endswith(".00"):
        x_clean = x_clean[:-3]
    try:
        return int(x_clean)
    except ValueError:
        return np.nan


def _parse_float(x):
    if pd.isna(x):
        return np.nan
    x_clean = str(x).strip().replace(",", "")
    try:
        return float(x_clean)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Month extraction from filenames
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _get_month_year_prime(file_path: str) -> pd.Timestamp:
    """Extract datetime from prime filename like ``cleaned_JUL_2025.csv``."""
    name = os.path.basename(file_path).upper()
    match = re.search(
        r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_(\d{4})", name,
    )
    if match:
        month = _MONTH_MAP[match.group(1)]
        year = int(match.group(2))
        return pd.Timestamp(year, month, 1)
    # fallback — very early date so it sorts first
    return pd.Timestamp("1900-01-01")


def _get_month_year_txn(file_path: str) -> pd.Timestamp:
    """Extract datetime from transaction filename like ``202506.xlsx``."""
    name = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r"(\d{4})(\d{2})", name)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return pd.Timestamp(year, month, 1)
    return pd.Timestamp("1900-01-01")


# ---------------------------------------------------------------------------
# Prime data
# ---------------------------------------------------------------------------

def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the prime data directory.

    - Reads with latin encoding and forces string dtypes on columns that
      contain commas or mixed types.
    - Renames ``RIM_NO:`` -> ``RIMNO``.
    - Parses numeric strings via _parse_int / _parse_float.
    - Drops columns not present in the current schema.
    - Tags each row with ``snapshot_month`` extracted from the filename.
    """
    data_dir = data_dir or config.PRIME_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*active.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Build dtype dict: read numeric-ish cols as string so we can clean them
    str_dtype = {
        col: "string"
        for col in (config.PRIME_STRING_COLS
                    + config.PRIME_INT_COLS
                    + config.PRIME_FLOAT_COLS)
    }

    frames = []
    def _read_csv_safe(f):
        # Read headers first to check if dates exist
        headers = pd.read_csv(f, nrows=0, encoding="latin").columns.tolist()
        parse_dates = [c for c in config.DATE_COLS_PRIME if c in headers]
        return pd.read_csv(
            f,
            encoding="latin",
            dtype=str_dtype,
            parse_dates=parse_dates,
        )

    for f in files:
        df = _read_csv_safe(f)
        df["source_file"] = os.path.basename(f)
        df[config.MONTH_COL] = _get_month_year_prime(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Rename customer ID column (raw data may have "RIM_NO" or "RIM_NO:")
    if "RIM_NO" in combined.columns:
        combined = combined.rename(columns={"RIM_NO": "RIMNO"})

    # Parse float columns (remove commas, cast)
    for col in config.PRIME_FLOAT_COLS:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_float)

    # Parse int columns
    for col in config.PRIME_INT_COLS:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_int)

    # Drop columns that no longer exist in the schema
    for col in ["MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT"]:
        if col in combined.columns:
            combined = combined.drop(columns=[col])

    # Re-parse date columns that may not have been auto-parsed
    for col in config.DATE_COLS_PRIME:
        if col in combined.columns and not pd.api.types.is_datetime64_any_dtype(combined[col]):
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    # Sort by month for deterministic ordering
    combined = combined.sort_values(config.MONTH_COL).reset_index(drop=True)

    print(f"[data_loader] Loaded {len(files)} prime file(s) -> {combined.shape}")
    months = combined[config.MONTH_COL].nunique()
    print(f"  Detected {months} unique month(s): "
          f"{sorted(combined[config.MONTH_COL].dt.strftime('%Y-%m').unique())}")
    return combined


# ---------------------------------------------------------------------------
# Transaction data
# ---------------------------------------------------------------------------

def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the transaction data directory.

    - Forces string dtypes on text columns.
    - Parses date columns.
    - Tags each row with ``snapshot_month`` extracted from the filename.
    """
    data_dir = data_dir or config.TRANSACTION_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    str_dtype = {col: "string" for col in config.TXN_STRING_COLS}

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin", dtype=str_dtype)
        df[config.MONTH_COL] = _get_month_year_txn(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Parse date columns
    for col in config.DATE_COLS_TXN:
        if col in combined.columns and not pd.api.types.is_datetime64_any_dtype(combined[col]):
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    print(f"[data_loader] Loaded {len(files)} transaction file(s) -> {combined.shape}")
    months = combined[config.MONTH_COL].nunique()
    print(f"  Detected {months} unique month(s): "
          f"{sorted(combined[config.MONTH_COL].dt.strftime('%Y-%m').unique())}")
    return combined


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_data(prime_df: pd.DataFrame, txn_features_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join prime data with aggregated transaction features.

    If both DataFrames contain a ``snapshot_month`` column, the merge is
    performed on ``(CUSTOMER_ID, snapshot_month)`` so that each monthly
    snapshot gets its own transaction summary. Otherwise falls back to a
    simple customer-ID merge.
    """
    cid = config.CUSTOMER_ID
    month_col = config.MONTH_COL
    txn_cols = txn_features_df.columns.tolist()

    if month_col in prime_df.columns and isinstance(txn_features_df.index, pd.MultiIndex):
        # Month-aware merge: txn_features is indexed by (CUSTOMER_ID, month)
        txn_reset = txn_features_df.reset_index()
        merged = prime_df.merge(txn_reset, on=[cid, month_col], how="left")
    else:
        # Fallback: original single-key merge
        merged = prime_df.merge(
            txn_features_df,
            left_on=cid,
            right_index=True,
            how="left",
        )

    # Fill NaN for customers with no transactions in that month
    merged[txn_cols] = merged[txn_cols].fillna(0)

    print(f"[data_loader] Merged shape: {merged.shape}")
    return merged
