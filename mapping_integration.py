"""mapping_integration.py

Adapter that integrates the existing scripts in `Credit-code/mapping/` into the
main ML pipeline without requiring you to rewrite those scripts.

The original scripts were written as standalone programs (top-level code that
runs on import and reads/writes files from fixed relative folders like
`prime/`, `transaction/`, `prime_cleaned/`, ...).

For pipeline integration we need functions that:
- work on in-memory DataFrames (prime_df, txn_df)
- don't read/write external folders
- are easy to disable/rollback

So this module re-implements the same *mapping intent* in a safe, minimal way:

1) Build a global numeric CUSTOMER_ID from prime data using (RIMNO, DOB)
   (same as `mapping/prime_id.py` PASS 1 concept).
2) Map transactions to CUSTOMER_ID using (RIMNO, PRODUCT_NAME)
   (same as `mapping/transaction_id_mapping.py`).

Outputs
-------
- Adds a new column `CUSTOMER_ID` (configurable) to both prime_df and txn_df.
- Keeps the original `RIMNO` column intact.

Important
---------
Your main pipeline currently aggregates transactions by `config.CUSTOMER_ID`.
To use mapping, we temporarily set `config.CUSTOMER_ID` to the mapped id column
*only inside the mapping call* OR we return the mapped id column name so the
caller can group by it.

This module chooses the simplest approach: it returns updated DataFrames and
also returns the mapped id column name.

If mapping quality is bad, disable it by setting config.ENABLE_MAPPING = False
(or simply not calling apply_mapping_layer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class MappingStats:
    prime_rows: int
    prime_customers_unique: int
    prime_mapped_rows: int
    txn_rows: int
    txn_mapped_rows: int
    txn_unmapped_rows: int


def _to_int64_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def build_customer_id_from_prime(prime_df: pd.DataFrame, *, rim_col: str = "RIMNO", dob_col: str = "DOB") -> pd.DataFrame:
    """Return a mapping DataFrame: (RIMNO, DOB) -> CUSTOMER_ID (1..N)."""
    if rim_col not in prime_df.columns:
        raise KeyError(f"Prime DF missing '{rim_col}'")
    if dob_col not in prime_df.columns:
        raise KeyError(f"Prime DF missing '{dob_col}' (required for mapping)")

    m = prime_df[[rim_col, dob_col]].copy()
    m[rim_col] = _to_int64_series(m[rim_col])
    m[dob_col] = pd.to_datetime(m[dob_col], errors="coerce")

    m = m.dropna(subset=[rim_col, dob_col]).drop_duplicates()
    m = m.sort_values([rim_col, dob_col]).reset_index(drop=True)
    m["CUSTOMER_ID"] = range(1, len(m) + 1)
    m["CUSTOMER_ID"] = _to_int64_series(m["CUSTOMER_ID"])
    return m


def apply_mapping_layer(
    prime_df: pd.DataFrame,
    txn_df: pd.DataFrame,
    *,
    rim_col: str = "RIMNO",
    dob_col: str = "DOB",
    txn_product_col: str = "DESCRIPTION",
    prime_product_col: str = "NAME",
    mapped_id_col: str = "CUSTOMER_ID",
) -> tuple[pd.DataFrame, pd.DataFrame, str, MappingStats]:
    """Apply mapping to prime + transactions.

    - Prime: CUSTOMER_ID assigned by (RIMNO, DOB)
    - Txn  : CUSTOMER_ID assigned by (RIMNO, PRODUCT_NAME)

    Returns: (prime_df2, txn_df2, mapped_id_col, stats)
    """
    p = prime_df.copy()
    t = txn_df.copy()

    # 1) Build global mapping from prime
    mapping_rim_dob = build_customer_id_from_prime(p, rim_col=rim_col, dob_col=dob_col)

    # 2) Attach CUSTOMER_ID to prime (left join on RIMNO, DOB)
    p[dob_col] = pd.to_datetime(p.get(dob_col), errors="coerce")
    p[rim_col] = _to_int64_series(p[rim_col])
    p = p.merge(mapping_rim_dob, on=[rim_col, dob_col], how="left")
    p = p.rename(columns={"CUSTOMER_ID": mapped_id_col})

    # 3) Build lookup for transactions: (RIMNO, PRODUCT_NAME) -> CUSTOMER_ID
    # Normalize product name to align with the script intent.
    # prime script used PRODUCT_NAME derived from NAME; txn script used DESCRIPTION.
    prime_products = p[[rim_col, prime_product_col, mapped_id_col]].copy()
    if prime_product_col in prime_products.columns:
        prime_products[prime_product_col] = prime_products[prime_product_col].astype("string").str.strip()
    prime_products[rim_col] = _to_int64_series(prime_products[rim_col])

    if txn_product_col in t.columns:
        t[txn_product_col] = t[txn_product_col].astype("string").str.strip()

    t[rim_col] = _to_int64_series(t[rim_col])

    # Build a small lookup with dedup. If duplicates exist, keep the first.
    lookup = (
        prime_products.dropna(subset=[rim_col, mapped_id_col])
        .drop_duplicates(subset=[rim_col, prime_product_col])
        .rename(columns={prime_product_col: txn_product_col})
    )

    t = t.merge(lookup[[rim_col, txn_product_col, mapped_id_col]], on=[rim_col, txn_product_col], how="left")

    stats = MappingStats(
        prime_rows=len(prime_df),
        prime_customers_unique=int(mapping_rim_dob[mapped_id_col if mapped_id_col in mapping_rim_dob.columns else "CUSTOMER_ID"].nunique()) if len(mapping_rim_dob) else 0,
        prime_mapped_rows=int(p[mapped_id_col].notna().sum()) if mapped_id_col in p.columns else 0,
        txn_rows=len(txn_df),
        txn_mapped_rows=int(t[mapped_id_col].notna().sum()) if mapped_id_col in t.columns else 0,
        txn_unmapped_rows=int(t[mapped_id_col].isna().sum()) if mapped_id_col in t.columns else len(txn_df),
    )

    print(
        f"[mapping] prime mapped rows: {stats.prime_mapped_rows:,}/{stats.prime_rows:,} | "
        f"txn mapped rows: {stats.txn_mapped_rows:,}/{stats.txn_rows:,} (unmapped: {stats.txn_unmapped_rows:,})"
    )

    return p, t, mapped_id_col, stats
