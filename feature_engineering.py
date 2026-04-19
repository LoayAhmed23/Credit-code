"""
Feature engineering for prime (customer snapshot) and transaction data.
Includes data cleaning steps (fill-na, outlier capping) derived from
the bank's actual data quirks.
"""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Prime features
# ---------------------------------------------------------------------------

def add_ratio_features(df):
    credit = df["CREDIT_LIMIT"].replace(0, np.nan)  # avoid division by zero

    # Core utilization — your strongest single feature
    df["utilization_ratio"] = df["LEDGER_BALANCE"] / credit

    # How far over the limit (0 for most customers, >0 is a red flag)
    # Note: We need to pull OVER_LIMIT here if it exists. Assume it's available or we compute it if Ledger > Credit limit. But from instructions, OVER_LIMIT seems to be a column name.
    # In the prompt, it says OVER_LIMIT. If it doesn't exist, we'll try to get it. Oh, looking at drop cols it says OVER_LIMIT.
    df["over_limit_ratio"] = df.get("OVER_LIMIT", df["LEDGER_BALANCE"] - df["CREDIT_LIMIT"]).clip(lower=0) / credit

    # Overdue severity — what fraction of the limit is unpaid past due date
    # In earlier config it says OVERDUEAMOUNT. Wait, instruction says OVERDUE_AMOUNT.
    df["overdue_ratio"] = df.get("OVERDUE_AMOUNT", df.get("OVERDUEAMOUNT", 0)) / credit

    # Payment behavior — are they paying the minimum or more?
    # >1 means they paid more than minimum (good), <1 means underpaying (bad)
    # instruction says MIN_PAYMENT, maybe it means MIN_PAYMENT_AMOUNT. Wait, in config I saw MIN_PAYMENT_AMOUNT.
    min_pay = df.get("MIN_PAYMENT", df.get("MIN_PAYMENT_AMOUNT", 0)).replace(0, np.nan)
    df["payment_coverage"] = df.get("LAST_PAYMENT_AMOUNT", 0) / min_pay

    # Composite stress score — useful as a single summary feature
    df["financial_stress_score"] = (
        df["utilization_ratio"] +
        df["over_limit_ratio"] * 2 +   # weighted heavier — over limit is worse
        df["overdue_ratio"] * 3         # weighted heaviest — overdue is the worst signal
    )

    return df

def engineer_prime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the prime (customer snapshot) data."""
    df = df.copy()
    extraction_date = pd.to_datetime("today")

    # --- Fill known missing-value patterns ---
    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].fillna("Unknown")
    if "ANNUAL_FEE" in df.columns:
        df["ANNUAL_FEE"] = pd.to_numeric(df["ANNUAL_FEE"], errors="coerce")
        df["ANNUAL_FEE"] = df["ANNUAL_FEE"].fillna(df["ANNUAL_FEE"].median())
    if "JOINING_FEE" in df.columns:
        df["JOINING_FEE"] = pd.to_numeric(df["JOINING_FEE"], errors="coerce")

    # --- Outlier capping (p99) for OVERDUEAMOUNT ---
    overdue_col = "OVERDUEAMOUNT"
    if overdue_col in df.columns:
        p99 = df[overdue_col].quantile(0.99)
        print(f"Overdue Outliers: {(df[overdue_col]>p99).sum()}")
        df[overdue_col] = np.where(df[overdue_col] > p99, p99, df[overdue_col])

    # --- Helper to safely extract and convert numeric columns ---
    def get_numeric_col(col_name: str) -> pd.Series:
        # If column exists, pd.to_numeric handles it returning a Series.
        # If not, we explicitly construct a Series of zeros.
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(0)
        else:
            return pd.Series(0, index=df.index, dtype=float)

    # --- Core numeric references ---
    credit_limit = get_numeric_col("CREDIT_LIMIT")
    available_limit = get_numeric_col("AVAILABLE_LIMIT")
    ledger = get_numeric_col("LEDGER_BALANCE")
    overdue = get_numeric_col("OVERDUEAMOUNT")
    total_hold = get_numeric_col("TOTAL_HOLD")
    last_payment = get_numeric_col("LAST_PAYMENT_AMOUNT")
    min_payment = get_numeric_col("MIN_PAYMENT")
    
    credit_limit_safe = credit_limit.replace({0: np.nan})
    min_payment_safe = min_payment.replace({0: np.nan})

    # --- Core utilization ---
    df["utilization_ratio"] = (ledger / credit_limit_safe).fillna(0)
    
    # --- Including pending transactions ---
    df["utilization_with_hold"] = ((ledger + total_hold) / credit_limit_safe).fillna(0)

    # --- Overdue severity ratio ---
    df["overdue_ratio"] = (overdue / credit_limit_safe).fillna(0)
    df["overdue_severity"] = np.where(
        credit_limit > 0,
        overdue / credit_limit,
        0,
    )

    # --- Minimum payment coverage ---
    df["payment_coverage"] = (last_payment / min_payment_safe).fillna(-1)

    # --- Available credit ratio ---
    df["available_credit_ratio"] = (available_limit / credit_limit_safe).fillna(0)

    # --- Fee to limit ratio ---
    joining_fee = pd.to_numeric(df.get("JOINING_FEE", 0), errors="coerce").fillna(0)
    annual_fee = pd.to_numeric(df.get("ANNUAL_FEE", 0), errors="coerce").fillna(0)
    df["fee_to_limit_ratio"] = ((joining_fee + annual_fee) / credit_limit_safe).fillna(0)

    # --- Card replacement count ---
    replacement_cols = [
        "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD",
    ]
    existing_rep_cols = [c for c in replacement_cols if c in df.columns]
    if existing_rep_cols:
        df["card_replacements"] = df[existing_rep_cols].notna().sum(axis=1)
    else:
        df["card_replacements"] = 0

    # --- Account tenure (months from creation to today) ---
    if "CREATION_DATE" in df.columns:
        df["account_tenure_months"] = (
            (extraction_date - df["CREATION_DATE"]).dt.days / 30
        ).fillna(0).clip(lower=0)
    else:
        df["account_tenure_months"] = 0

    # --- Days since last payment (from today) ---
    if "LAST_PAYMENT_DATE" in df.columns:
        df["days_since_last_payment"] = (
            (extraction_date - df["LAST_PAYMENT_DATE"]).dt.days
        ).fillna(-1)
    else:
        df["days_since_last_payment"] = -1

    # --- Time to churn (closure - creation) ---
    if "CLOSURE_DATE" in df.columns and "CREATION_DATE" in df.columns:
        df["time_to_churn_days"] = (
            (df["CLOSURE_DATE"] - df["CREATION_DATE"]).dt.days
        ).fillna(-1)
    else:
        df["time_to_churn_days"] = -1

    # --- Active flag ---
    if "ACTIVATED" in df.columns:
        df["is_active"] = (df["ACTIVATED"].astype(str).str.strip().str.upper() == "A").astype(int)
    else:
        df["is_active"] = 1

    df = add_ratio_features(df)

    # --- Customer age ---
    if "DOB" in df.columns:
        df["customer_age"] = (
            (extraction_date - df["DOB"]).dt.days / 365
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
    df["ledger_to_limit_ratio"] = np.where(
        credit_limit > 0,
        ledger / credit_limit,
        0,
    )

    # --- Additional ratio features ---
    df = add_ratio_features(df)

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
    cid = config.CUSTOMER_ID
    extraction_date = pd.to_datetime("today")

    # Ensure numeric amounts
    for col in ["BILLING AMT", "ORIG AMOUNT", "SETTLEMENT AMT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Outlier capping at p99 for amount columns ---
    for col in ["BILLING AMT", "SETTLEMENT AMT", "ORIG AMOUNT"]:
        if col in df.columns:
            p99 = df[col].quantile(0.99)
            print(f"{col} Outliers: {(df[col]>p99).sum()}")
            df[col] = np.where(df[col] > p99, p99, df[col])

    # --- Fill ORIG AMOUNT NaN with median ---
    if "ORIG AMOUNT" in df.columns:
        df["ORIG AMOUNT"] = df["ORIG AMOUNT"].fillna(df["ORIG AMOUNT"].median())

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

    # --- Days since last transaction (recency) ---
    if "TRXN DATE" in df.columns:
        txn_recency = df.groupby(cid)["TRXN DATE"].max()
        agg["days_since_last_txn"] = (extraction_date - txn_recency).dt.days.fillna(-1)
    else:
        agg["days_since_last_txn"] = -1

    print(f"[feature_eng] Transaction features -> {agg.shape}")
    return agg


# ---------------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------------

def create_target(df: pd.DataFrame, mode: str=config.TARGET_MODE) -> pd.Series:
    """Create binary default target from STATUS column.

    1 = default  (30DD, 60DA, 90DA, SUSP, WROF)
    0 = non-default (everything else)
    """
    mode = getattr(config, "TARGET_MODE", "all")

    if mode == "hard_only":
        df["target"] = df[config.STATUS_COL].isin(config.HARD_DEFAULT_STATUSES).astype(int)

    elif mode == "tiered":
        df["target"] = np.where(
            df[config.STATUS_COL].isin(config.HARD_DEFAULT_STATUSES), 2,
            np.where(df[config.STATUS_COL].isin(config.SOFT_DEFAULT_STATUSES), 1, 0)
        )

    elif mode == "weighted_binary":
        df["target"] = df[config.STATUS_COL].isin(
            config.HARD_DEFAULT_STATUSES + config.SOFT_DEFAULT_STATUSES
        ).astype(int)
        
        df["sample_weight"] = df[config.STATUS_COL].map({
            **{s: 1.0 for s in config.HARD_DEFAULT_STATUSES},
            **{s: 0.4 for s in config.SOFT_DEFAULT_STATUSES},
        }).fillna(0.0)

    elif mode == "all":
        df["target"] = df[config.STATUS_COL].isin(config.DEFAULT_STATUSES).astype(int)

    return df["target"]
