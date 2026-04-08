"""
Feature engineering for prime (customer snapshot) and transaction data.
"""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Prime features
# ---------------------------------------------------------------------------

def engineer_prime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the prime (customer snapshot) data."""
    df = df.copy()

    # --- Utilization ratio ---
    credit_limit = pd.to_numeric(df.get("CREDIT LIMIT", 0), errors="coerce").fillna(0)
    available_limit = pd.to_numeric(df.get("AVAILABLE_LIMIT", 0), errors="coerce").fillna(0)
    df["utilization_ratio"] = np.where(
        credit_limit > 0,
        (credit_limit - available_limit) / credit_limit,
        0,
    )

    # --- Payment-to-minimum-due ratio ---
    last_payment = pd.to_numeric(df.get("LAST_PAYMENT_AMOUNT", 0), errors="coerce").fillna(0)
    min_payment = pd.to_numeric(df.get("MIN_PAYMENT", 0), errors="coerce").fillna(0)
    df["payment_to_min_ratio"] = np.where(
        min_payment > 0,
        (last_payment / min_payment).clip(upper=10),
        0,
    )

    # --- Overdue severity ---
    overdue = pd.to_numeric(df.get("OVERDUE_AMOUNT", 0), errors="coerce").fillna(0)
    df["overdue_severity"] = np.where(
        credit_limit > 0,
        overdue / credit_limit,
        0,
    )

    # --- Over-limit flag ---
    over_limit = pd.to_numeric(df.get("OVER_LIMIT", 0), errors="coerce").fillna(0)
    df["overlimit_flag"] = (over_limit > 0).astype(int)

    # --- Card replacement count ---
    replacement_cols = [
        "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD",
    ]
    existing_rep_cols = [c for c in replacement_cols if c in df.columns]
    if existing_rep_cols:
        df["card_replacements"] = df[existing_rep_cols].notna().sum(axis=1)
    else:
        df["card_replacements"] = 0

    # --- Account age (days) ---
    if "LAST_STATEMENT_DATE" in df.columns and "CREATION_DATE" in df.columns:
        df["account_age_days"] = (
            (df["LAST_STATEMENT_DATE"] - df["CREATION_DATE"]).dt.days
        ).fillna(0).clip(lower=0)
    else:
        df["account_age_days"] = 0

    # --- Days since last payment ---
    if "LAST_STATEMENT_DATE" in df.columns and "LAST_PAYMENT_DATE" in df.columns:
        df["days_since_last_payment"] = (
            (df["LAST_STATEMENT_DATE"] - df["LAST_PAYMENT_DATE"]).dt.days
        ).fillna(-1)
    else:
        df["days_since_last_payment"] = -1

    # --- Active flag ---
    if "ACTIVATED" in df.columns:
        df["is_active"] = (df["ACTIVATED"].astype(str).str.strip().str.upper() == "A").astype(int)
    else:
        df["is_active"] = 1

    # --- Customer age ---
    if "DOB" in df.columns and "LAST_STATEMENT_DATE" in df.columns:
        df["customer_age"] = (
            (df["LAST_STATEMENT_DATE"] - df["DOB"]).dt.days / 365.25
        ).fillna(0).clip(lower=0)
    else:
        df["customer_age"] = 0

    # --- Primary card flag ---
    if "CUSTOMER_TYPE" in df.columns:
        df["is_primary"] = (
            df["CUSTOMER_TYPE"].astype(str).str.strip().str.upper().str.contains("PRIM")
        ).astype(int)
    else:
        df["is_primary"] = 1

    # --- Ledger balance to limit ratio ---
    ledger = pd.to_numeric(df.get("LEDGER_BALANCE", 0), errors="coerce").fillna(0)
    df["ledger_to_limit_ratio"] = np.where(
        credit_limit > 0,
        ledger / credit_limit,
        0,
    )

    print(f"[feature_eng] Prime features engineered -> {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Transaction features
# ---------------------------------------------------------------------------

def engineer_transaction_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into per-customer features.

    Returns a DataFrame indexed by customer ID (RIMNO).
    """
    df = txn_df.copy()
    cid = config.TXN_CUSTOMER_ID

    # Ensure numeric amounts
    for col in ["BILLING AMT", "ORIG AMOUNT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Basic transaction aggregations
    agg = df.groupby(cid).agg(
        txn_count=(cid, "size"),
        txn_total_amount=("BILLING AMT", "sum"),
        txn_mean_amount=("BILLING AMT", "mean"),
        txn_max_amount=("BILLING AMT", "max"),
    )

    # International transaction count (CCY != 818 for non-EGP)
    if "CCY" in df.columns:
        df["_is_intl"] = (pd.to_numeric(df["CCY"], errors="coerce") != 818).astype(int)
        intl = df.groupby(cid)["_is_intl"].agg(
            txn_intl_count="sum",
        )
        agg = agg.join(intl)
        agg["txn_intl_ratio"] = np.where(
            agg["txn_count"] > 0,
            agg["txn_intl_count"] / agg["txn_count"],
            0,
        )
    else:
        agg["txn_intl_count"] = 0
        agg["txn_intl_ratio"] = 0

    # Reversal count
    if "REVERSAL FLAG" in df.columns:
        df["_is_reversal"] = (
            df["REVERSAL FLAG"].astype(str).str.strip().str.upper().isin(["Y", "YES", "1", "TRUE"])
        ).astype(int)
        rev = df.groupby(cid)["_is_reversal"].agg(txn_reversal_count="sum")
        agg = agg.join(rev)
        agg["txn_reversal_ratio"] = np.where(
            agg["txn_count"] > 0,
            agg["txn_reversal_count"] / agg["txn_count"],
            0,
        )
    else:
        agg["txn_reversal_count"] = 0
        agg["txn_reversal_ratio"] = 0

    # Merchant diversity
    if "MERCH ID" in df.columns:
        agg = agg.join(df.groupby(cid)["MERCH ID"].nunique().rename("unique_merchants"))
    else:
        agg["unique_merchants"] = 0

    if "MCC" in df.columns:
        agg = agg.join(df.groupby(cid)["MCC"].nunique().rename("unique_mcc"))
    else:
        agg["unique_mcc"] = 0

    # Payment success rate from PaymentBehaviourHistory
    if "PaymentBehaviourHistory" in df.columns:
        def _success_rate(series):
            combined = "".join(series.dropna().astype(str))
            if not combined:
                return 0
            return combined.count("T") / len(combined) if len(combined) > 0 else 0

        psr = df.groupby(cid)["PaymentBehaviourHistory"].apply(_success_rate).rename(
            "payment_success_rate"
        )
        agg = agg.join(psr)
    else:
        agg["payment_success_rate"] = 0

    print(f"[feature_eng] Transaction features -> {agg.shape}")
    return agg


# ---------------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------------

def create_target(df: pd.DataFrame) -> pd.Series:
    """Create binary default target from STATUS column.

    1 = default  (30DD, 90DA, SUSP, WROF)
    0 = non-default (NORM, CLSB, CLSC)
    """
    status = df[config.STATUS_COL].astype(str).str.strip().str.upper()
    target = status.map(
        {s: 1 for s in config.DEFAULT_STATUSES}
        | {s: 0 for s in config.NON_DEFAULT_STATUSES}
    )
    unmapped = target.isna().sum()
    if unmapped > 0:
        print(f"[feature_eng] WARNING: {unmapped} rows with unknown STATUS -> dropped")
        target = target.dropna()
    return target.astype(int).rename(config.TARGET_COL)
