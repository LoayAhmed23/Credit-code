"""
Data loading utilities.
Reads monthly CSVs from prime_data/ and transaction_data/ folders,
parses dates, and provides a merge helper.
"""

import glob
import os

import pandas as pd

import config


def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the prime data directory."""
    data_dir = data_dir or config.PRIME_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Parse date columns (coerce errors so bad values become NaT)
    for col in config.DATE_COLS_PRIME:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    print(f"[data_loader] Loaded {len(files)} prime file(s) -> {combined.shape}")
    return combined


def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the transaction data directory."""
    data_dir = data_dir or config.TRANSACTION_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No XLSX files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_excel(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Parse date columns
    for col in config.DATE_COLS_TXN:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    print(f"[data_loader] Loaded {len(files)} transaction file(s) -> {combined.shape}")
    return combined


def merge_data(prime_df: pd.DataFrame, txn_features_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join prime data with aggregated transaction features on customer ID."""
    merged = prime_df.merge(
        txn_features_df,
        left_on=config.PRIME_CUSTOMER_ID,
        right_index=True,   
        how="left",
    )
    # Fill NaN for customers with no transactions
    txn_cols = txn_features_df.columns.tolist()
    merged[txn_cols] = merged[txn_cols].fillna(0)

    print(f"[data_loader] Merged shape: {merged.shape}")
    return merged
