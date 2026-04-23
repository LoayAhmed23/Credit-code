"""
Data loading utilities.
Reads monthly CSVs from prime/ and transaction/ directories,
enforces dtypes via vectorised casting, cleans columns, and provides
a merge helper.

Preprocessing approach adapted from data_to_to_apply.py.
"""

import glob
import os

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Vectorised cast-and-report (from data_to_to_apply.py)
# ---------------------------------------------------------------------------

def apply_cast_and_report(df: pd.DataFrame, columns: list, cast_type: str):
    """Cast *columns* to *cast_type* in-place and report null changes.

    Supported cast_type values: 'string', 'float', 'int', 'date'.
    """
    total_rows = len(df)

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue

        nulls_before = df[col].isna().sum()

        if cast_type == "string":
            df[col] = df[col].astype("string")

        elif cast_type == "float":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif cast_type == "int":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        elif cast_type == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")

        nulls_after = df[col].isna().sum()
        pct_missing = (nulls_after / total_rows) * 100
        print(
            f"[{col}] Type: {df[col].dtype} | "
            f"Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}% missing)"
        )


def drop_empty_records(df: pd.DataFrame, columns):
    """Drop rows missing data in *columns* and print a summary."""
    if isinstance(columns, str):
        columns = [columns]

    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        print(f"\nWarning: None of {columns} exist in the dataframe. No rows dropped.")
        return df

    rows_before = len(df)
    cleaned_df = df.dropna(subset=valid_cols, how="any")
    rows_after = len(cleaned_df)

    print(f"\n--- Dropping Empty Records ---")
    print(f"Checked columns: {valid_cols}")
    print(f"Dropped {rows_before - rows_after} rows.")
    print(f"Total rows remaining: {rows_after}")

    return cleaned_df


# ---------------------------------------------------------------------------
# Column-type definitions (from data_to_to_apply.py)
# ---------------------------------------------------------------------------

PRIME_STRING_COLS = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
    "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status ",
]
PRIME_INT_COLS = ["BRANCH_ID", "RIMNO"]
PRIME_FLOAT_COLS = [
    "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD", "SETTLEMENT AMT",
]
PRIME_DATE_COLS = [
    "CREATION_DATE", "LAST_STAEMENT_DATE", "LAST_PAYMENT_DATE",
    "DOB", "CLOSURE_DATE",
]

PRIME_COLUMNS_TO_DROP = [
    "MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT", "TOTAL_HOLD",
    "ORGANIZATION", "JOINING_FEE", "ANNUAL_FEE",
    "LAST_PAYMENT_AMOUNT", "LAST_PAYMENT_DATE", "SETTLEMENT AMT",
]

TXN_STRING_COLS = [
    "DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES",
    "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME",
]
TXN_INT_COLS = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY"]
TXN_FLOAT_COLS = ["ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]
TXN_DATE_COLS = ["TRXN DATE", "POST DATE"]

TXN_COLUMNS_TO_DROP = [
    "POST DATE", "ORIG AMOUNT", "EMBEDDED _FEE",
    "SETTLEMENT AMT", "SETTLEMENT CCY", "SOURCES",
]


# ---------------------------------------------------------------------------
# Prime data
# ---------------------------------------------------------------------------

def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the prime data directory.

    Uses the casting approach from data_to_to_apply.py:
    - Reads every column as string initially
    - Applies vectorised casting via apply_cast_and_report
    - Drops unnecessary columns
    - Drops rows with missing RIMNO
    """
    data_dir = data_dir or config.PRIME_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_str_cols = PRIME_STRING_COLS + PRIME_INT_COLS + PRIME_FLOAT_COLS
    str_dtype = {col: "string" for col in all_str_cols}

    frames = []
    print("Loading prime CSV files...")
    for f in files:
        df = pd.read_csv(
            f,
            encoding="latin",
            dtype=str_dtype,
            parse_dates=PRIME_DATE_COLS,
        ).rename(columns={"RIM_NO": "RIMNO", "NAME": "PRODUCT_NAME"})
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # --- Casting & reporting ---
    print("\n--- Casting Prime Columns and Checking Nulls ---")
    print("\n-> String Columns:")
    apply_cast_and_report(combined, PRIME_STRING_COLS, "string")
    print("\n-> Float Columns:")
    apply_cast_and_report(combined, PRIME_FLOAT_COLS, "float")
    print("\n-> Integer Columns:")
    apply_cast_and_report(combined, PRIME_INT_COLS, "int")
    print("\n-> Date Columns:")
    apply_cast_and_report(combined, PRIME_DATE_COLS, "date")

    # --- Drop unnecessary columns ---
    existing_drop = [c for c in PRIME_COLUMNS_TO_DROP if c in combined.columns]
    combined = combined.drop(columns=existing_drop)

    # --- Drop rows with missing RIMNO ---
    combined = drop_empty_records(combined, ["RIMNO"])

    print(f"\n[data_loader] Loaded {len(files)} prime file(s) -> {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Transaction data
# ---------------------------------------------------------------------------

def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all CSV files from the transaction data directory.

    Uses the casting approach from data_to_to_apply.py.
    """
    data_dir = data_dir or config.TRANSACTION_DATA_DIR
    # Support both CSV and XLSX; prefer CSV
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    xlsx_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    files = csv_files or xlsx_files
    if not files:
        raise FileNotFoundError(
            f"No CSV or XLSX files found in {data_dir}"
        )

    all_str_cols = TXN_STRING_COLS + TXN_INT_COLS + TXN_FLOAT_COLS
    str_dtype = {col: "string" for col in all_str_cols}

    frames = []
    print("\nLoading Transaction files...")
    for f in files:
        if f.endswith(".csv"):
            df = pd.read_csv(
                f,
                encoding="latin",
                dtype=str_dtype,
                parse_dates=TXN_DATE_COLS,
            )
        else:
            df = pd.read_excel(f, dtype=str_dtype)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # --- Casting & reporting ---
    print("\n--- Casting Transaction Columns and Checking Nulls ---")
    print("\n-> Transaction String Columns:")
    apply_cast_and_report(combined, TXN_STRING_COLS, "string")
    print("\n-> Transaction Float Columns:")
    apply_cast_and_report(combined, TXN_FLOAT_COLS, "float")
    print("\n-> Transaction Integer Columns:")
    apply_cast_and_report(combined, TXN_INT_COLS, "int")
    print("\n-> Transaction Date Columns:")
    apply_cast_and_report(combined, TXN_DATE_COLS, "date")

    # --- Drop unnecessary columns ---
    existing_drop = [c for c in TXN_COLUMNS_TO_DROP if c in combined.columns]
    combined = combined.drop(columns=existing_drop)

    print(f"\n[data_loader] Loaded {len(files)} transaction file(s) -> {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Filter: keep only RIMNOs that have transactions
# ---------------------------------------------------------------------------

def filter_rimno_with_transactions(
    prime_df: pd.DataFrame,
    txn_df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows from *prime_df* whose RIMNO has zero transactions.

    This ensures the model only trains on / scores customers that have
    observable transaction behaviour.
    """
    cid = config.CUSTOMER_ID
    rimno_with_txn = txn_df[cid].dropna().unique()

    before = len(prime_df)
    prime_df = prime_df[prime_df[cid].isin(rimno_with_txn)]
    after = len(prime_df)

    print(
        f"[data_loader] Filtered RIMNOs with no transactions: "
        f"dropped {before - after} rows, {after} remaining."
    )
    return prime_df


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
