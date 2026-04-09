"""
Data loading utilities.
Reads monthly CSVs from prime_data/ and transaction XLSX files from
transaction_data/, enforces dtypes, cleans numeric strings, parses dates,
and provides a merge helper.
"""

import glob
import os

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
# Prime data
# ---------------------------------------------------------------------------

def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the prime data directory.

    - Reads with latin encoding and forces string dtypes on columns that
      contain commas or mixed types.
    - Renames ``RIM_NO:`` -> ``RIMNO``.
    - Parses numeric strings via _parse_int / _parse_float.
    - Drops columns not present in the current schema.
    """
    data_dir = data_dir or config.PRIME_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
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
    for f in files:
        df = pd.read_csv(
            f,
            encoding="latin",
            dtype=str_dtype,
            parse_dates=config.DATE_COLS_PRIME,
        )
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Rename customer ID column (raw data may have "RIM_NO:" or "RIM_NO")
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

    print(f"[data_loader] Loaded {len(files)} prime file(s) -> {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Transaction data
# ---------------------------------------------------------------------------

def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all XLSX files from the transaction data directory.

    - Reads with latin encoding, forces string dtypes on text columns.
    - Parses date columns.
    """
    data_dir = data_dir or config.TRANSACTION_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No XLSX files found in {data_dir}")

    str_dtype = {col: "string" for col in config.TXN_STRING_COLS}

    frames = []
    for f in files:
        df = pd.read_excel(f, dtype=str_dtype)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Parse date columns
    for col in config.DATE_COLS_TXN:
        if col in combined.columns and not pd.api.types.is_datetime64_any_dtype(combined[col]):
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    print(f"[data_loader] Loaded {len(files)} transaction file(s) -> {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_data(prime_df: pd.DataFrame, txn_features_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join prime data with aggregated transaction features on customer ID."""
    merged = prime_df.merge(
        txn_features_df,
        left_on=config.CUSTOMER_ID,
        right_index=True,
        how="left",
    )
    # Fill NaN for customers with no transactions
    txn_cols = txn_features_df.columns.tolist()
    merged[txn_cols] = merged[txn_cols].fillna(0)

    print(f"[data_loader] Merged shape: {merged.shape}")
    return merged
