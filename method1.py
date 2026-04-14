"""
churn_labeller.py
-----------------
Adds a CHURN column to the latest prime snapshot.

Logic
-----
1.  Sort all prime CSVs chronologically by filename; the last one is
    "latest_prime", the rest are "historical".
2.  Collect every RIMNO that has ever appeared in the historical files.
3.  For each historical RIMNO, look it up in latest_prime:
      - RIMNO exists   → CHURN = 1 if *every* row for that RIMNO has a
                         status in CHURN_STATUSES, else 0.
      - RIMNO missing  → append a stub row with CHURN = 1.
4.  RIMNOs that only appear in latest_prime (never seen before) get
    CHURN = 0 by default.
5.  Writes the result to OUTPUT_PATH.

Adjust CHURN_STATUSES and PRIME_DATA_DIR as needed.
"""

import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

PRIME_DATA_DIR = "prime_data"          # folder containing all prime CSVs
OUTPUT_PATH    = "output/latest_prime_with_churn.csv"

# Statuses that, when held by ALL rows of a RIMNO in the latest snapshot,
# indicate the customer has churned.
CHURN_STATUSES = {
    "CLSB",   # closed by bank
    "CLSC",   # closed by customer
    "CLSD",   # closed
    "WROF",   # write-off
    "EXPD",   # expired (account)
    "EXPC",   # expired card
    "LOST",   # lost card (card no longer usable)
    "STLC",   # stolen card
    "FRAD",   # fraud pick-up
    "PICK",   # pick-up card
}

# Column names
CUSTOMER_ID_COL = "RIMNO"          # after rename from RIM_NO
STATUS_COL      = "STATUS"
CHURN_COL       = "CHURN"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """Rename RIM_NO → RIMNO if needed."""
    if "RIM_NO" in df.columns and CUSTOMER_ID_COL not in df.columns:
        df = df.rename(columns={"RIM_NO": CUSTOMER_ID_COL})
    return df


def _load_csv(path: str) -> pd.DataFrame:
    """Read a single prime CSV with robust encoding."""
    df = pd.read_csv(path, encoding="latin", dtype=str, low_memory=False)
    df = _normalise_id_col(df)
    # Normalise STATUS to uppercase, strip whitespace
    if STATUS_COL in df.columns:
        df[STATUS_COL] = df[STATUS_COL].astype(str).str.strip().str.upper()
    # Normalise RIMNO
    if CUSTOMER_ID_COL in df.columns:
        df[CUSTOMER_ID_COL] = (
            df[CUSTOMER_ID_COL].astype(str).str.strip().str.replace(",", "")
        )
    return df


def _is_churned_status(status: str) -> bool:
    return status.upper().strip() in CHURN_STATUSES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def label_churn():
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # 1. Discover and sort all prime CSVs
    all_files = sorted(glob.glob(os.path.join(PRIME_DATA_DIR, "*.csv")))
    if len(all_files) < 2:
        raise ValueError(
            f"Need at least 2 prime CSV files in '{PRIME_DATA_DIR}'; "
            f"found {len(all_files)}."
        )

    latest_file    = all_files[-1]
    history_files  = all_files[:-1]

    print(f"Latest prime file   : {os.path.basename(latest_file)}")
    print(f"Historical files    : {[os.path.basename(f) for f in history_files]}")
    print()

    # 2. Collect every RIMNO seen in historical files
    historical_rimnos: set[str] = set()
    for f in history_files:
        df = _load_csv(f)
        if CUSTOMER_ID_COL in df.columns:
            historical_rimnos.update(df[CUSTOMER_ID_COL].dropna().unique())

    print(f"Unique RIMNOs in historical files: {len(historical_rimnos):,}")

    # 3. Load latest prime
    latest = _load_csv(latest_file)
    print(f"Rows in latest prime             : {len(latest):,}")

    # Initialise CHURN column to 0 for everyone already in latest_prime
    latest[CHURN_COL] = 0

    # 4. For each historical RIMNO, decide CHURN
    rimnos_in_latest = set(latest[CUSTOMER_ID_COL].dropna().unique())

    # 4a. RIMNOs that exist in latest_prime
    rimnos_in_both = historical_rimnos & rimnos_in_latest

    # Vectorised: for each RIMNO in latest, flag rows that belong to
    # customers whose EVERY status is a churn status.
    # Group by RIMNO, check if all statuses are churn statuses.
    if STATUS_COL in latest.columns:
        churn_mask_per_rimno = (
            latest[latest[CUSTOMER_ID_COL].isin(rimnos_in_both)]
            .groupby(CUSTOMER_ID_COL)[STATUS_COL]
            .apply(lambda statuses: statuses.map(_is_churned_status).all())
        )
        # churn_mask_per_rimno is a Series: RIMNO -> True/False
        churned_rimnos = set(
            churn_mask_per_rimno[churn_mask_per_rimno].index.tolist()
        )
    else:
        print(f"WARNING: '{STATUS_COL}' column not found in latest prime — "
              "all existing customers set to CHURN=0.")
        churned_rimnos = set()

    # Apply CHURN=1 to rows in latest whose RIMNO is fully churned
    latest.loc[latest[CUSTOMER_ID_COL].isin(churned_rimnos), CHURN_COL] = 1

    print(f"  Historical RIMNOs found in latest   : {len(rimnos_in_both):,}")
    print(f"  Of those, all rows churned (CHURN=1): {len(churned_rimnos):,}")

    # 4b. RIMNOs that are in historical files but MISSING from latest_prime
    rimnos_missing = historical_rimnos - rimnos_in_latest
    print(f"  Historical RIMNOs missing from latest: {len(rimnos_missing):,}")

    if rimnos_missing:
        # Build stub rows: one row per missing RIMNO, all columns NaN
        # except RIMNO and CHURN.
        stub_rows = pd.DataFrame(
            {CUSTOMER_ID_COL: list(rimnos_missing)}
        )
        stub_rows[CHURN_COL] = 1

        # Add any columns present in latest but absent in stub (fill NaN)
        for col in latest.columns:
            if col not in stub_rows.columns:
                stub_rows[col] = np.nan

        # Re-order columns to match latest
        stub_rows = stub_rows[latest.columns]

        latest = pd.concat([latest, stub_rows], ignore_index=True)
        print(f"  Stub rows appended for missing RIMNOs: {len(stub_rows):,}")

    # 5. Summary
    total_churn    = int(latest[CHURN_COL].sum())
    total_rows     = len(latest)
    total_non_churn = total_rows - total_churn
    print()
    print("=" * 50)
    print("CHURN SUMMARY")
    print("=" * 50)
    print(f"  Total rows  : {total_rows:,}")
    print(f"  CHURN = 1   : {total_churn:,}  ({total_churn / total_rows * 100:.1f}%)")
    print(f"  CHURN = 0   : {total_non_churn:,}  ({total_non_churn / total_rows * 100:.1f}%)")
    print()

    # 6. Write output
    latest.to_csv(OUTPUT_PATH, index=False)
    print(f"Output saved to: {OUTPUT_PATH}")

    return latest


if __name__ == "__main__":
    result = label_churn()