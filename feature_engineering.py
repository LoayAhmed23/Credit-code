"""
Feature engineering for prime (customer snapshot) and transaction data.
Includes data cleaning steps (fill-na, outlier capping) derived from
the bank's actual data quirks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Columns that directly encode the target and must never become features.
# This list is checked at the end of engineering as a safety net — the
# primary removal happens via DROP_COLS in preprocessing, but an explicit
# guard here catches bugs earlier and gives a clearer error message.
# ---------------------------------------------------------------------------
_LEAKAGE_COLS = {
    "STATUS", "STATUS_NAME", "DELINQUENCY",
    "CARD_ACCOUNT_STATUS", "Card account status ",
    "CLOSURE_DATE",          # non-null only for defaulted/closed accounts
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_numeric(df: pd.DataFrame, *col_names: str, default: float = 0.0) -> pd.Series:
    """Return the first matching column as a numeric Series, or a zero Series."""
    for col in col_names:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _assert_no_leakage(df: pd.DataFrame, stage: str) -> None:
    """Raise if any known leakage column survived into the feature DataFrame."""
    found = _LEAKAGE_COLS.intersection(df.columns)
    if found:
        raise ValueError(
            f"[leakage] The following columns must be dropped before '{stage}' "
            f"but are still present: {sorted(found)}\n"
            f"Add them to config.DROP_COLS and re-run."
        )


# ---------------------------------------------------------------------------
# Prime features
# ---------------------------------------------------------------------------

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

    # --- Outlier capping (p99) for overdue amount ---
    overdue_col = "OVERDUEAMOUNT" if "OVERDUEAMOUNT" in df.columns else "OVERDUE_AMOUNT"
    if overdue_col in df.columns:
        p99 = df[overdue_col].quantile(0.99)
        print(f"[feature_eng] Overdue outliers capped at {p99:.2f}: "
              f"{(df[overdue_col] > p99).sum()} rows")
        df[overdue_col] = df[overdue_col].clip(upper=p99)

    # --- Core numeric references (accept both naming conventions) ---
    credit_limit    = _get_numeric(df, "CREDIT_LIMIT")
    available_limit = _get_numeric(df, "AVAILABLE_LIMIT")
    ledger          = _get_numeric(df, "LEDGER_BALANCE")
    overdue         = _get_numeric(df, "OVERDUEAMOUNT", "OVERDUE_AMOUNT")
    total_hold      = _get_numeric(df, "TOTAL_HOLD")
    last_payment    = _get_numeric(df, "LAST_PAYMENT_AMOUNT")
    min_payment     = _get_numeric(df, "MIN_PAYMENT", "MIN_PAYMENT_AMOUNT")
    over_limit_raw  = _get_numeric(df, "OVER_LIMIT")

    credit_limit_safe = credit_limit.replace(0, np.nan)
    min_payment_safe  = min_payment.replace(0, np.nan)

    # --- Utilization ratios ---
    df["utilization_ratio"]      = (ledger / credit_limit_safe).fillna(0)
    df["utilization_with_hold"]  = ((ledger + total_hold) / credit_limit_safe).fillna(0)
    df["available_credit_ratio"] = (available_limit / credit_limit_safe).fillna(0)

    # --- Over-limit ratio ---
    # Use the dedicated column if it exists; otherwise derive it.
    if "OVER_LIMIT" in df.columns:
        over_limit_val = over_limit_raw
    else:
        over_limit_val = (ledger - credit_limit).clip(lower=0)
    df["over_limit_ratio"] = (over_limit_val / credit_limit_safe).fillna(0)

    # --- Overdue ratio ---
    # NOTE: we do NOT create an "overdue_severity" alias — it was identical
    # to overdue_ratio and the duplicate confused feature importance charts.
    df["overdue_ratio"] = (overdue / credit_limit_safe).fillna(0)

    # --- Payment coverage (>1 = paid more than minimum, <1 = underpaying) ---
    df["payment_coverage"] = (last_payment / min_payment_safe).fillna(-1)

    # --- Composite financial stress score ---
    # Weights reflect increasing severity: utilization < over-limit < overdue.
    df["financial_stress_score"] = (
        df["utilization_ratio"]
        + df["over_limit_ratio"] * 2
        + df["overdue_ratio"]    * 3
    )

    # --- Fee-to-limit ratio ---
    joining_fee = _get_numeric(df, "JOINING_FEE")
    annual_fee  = _get_numeric(df, "ANNUAL_FEE")
    df["fee_to_limit_ratio"] = ((joining_fee + annual_fee) / credit_limit_safe).fillna(0)

    # --- Card replacement count ---
    replacement_cols = [
        "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD",
    ]
    existing_rep = [c for c in replacement_cols if c in df.columns]
    df["card_replacements"] = df[existing_rep].notna().sum(axis=1) if existing_rep else 0

    # --- Account tenure (months since creation) ---
    if "CREATION_DATE" in df.columns:
        df["account_tenure_months"] = (
            (extraction_date - df["CREATION_DATE"]).dt.days / 30
        ).fillna(0).clip(lower=0)
    else:
        df["account_tenure_months"] = 0

    # --- Days since last payment ---
    if "LAST_PAYMENT_DATE" in df.columns:
        df["days_since_last_payment"] = (
            (extraction_date - df["LAST_PAYMENT_DATE"]).dt.days
        ).fillna(-1)
    else:
        df["days_since_last_payment"] = -1

    # --- Time to churn (closure - creation) ---
    # IMPORTANT: this feature is derived from CLOSURE_DATE which is a leakage
    # source (non-null only for closed/defaulted accounts). The raw
    # CLOSURE_DATE is in DROP_COLS; time_to_churn_days is also dropped there
    # because it encodes the same information.
    if "CLOSURE_DATE" in df.columns and "CREATION_DATE" in df.columns:
        df["time_to_churn_days"] = (
            (df["CLOSURE_DATE"] - df["CREATION_DATE"]).dt.days
        ).fillna(-1)
    else:
        df["time_to_churn_days"] = -1

    # --- Active flag ---
    if "ACTIVATED" in df.columns:
        df["is_active"] = (
            df["ACTIVATED"].astype(str).str.strip().str.upper() == "A"
        ).astype(int)
    else:
        df["is_active"] = 1

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

    # --- Ensure numeric amounts ---
    for col in ["BILLING AMT", "ORIG AMOUNT", "SETTLEMENT AMT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Outlier capping at p99 ---
    for col in ["BILLING AMT", "SETTLEMENT AMT", "ORIG AMOUNT"]:
        if col in df.columns:
            p99 = df[col].quantile(0.99)
            print(f"[feature_eng] {col} outliers capped at {p99:.2f}: "
                  f"{(df[col] > p99).sum()} rows")
            df[col] = df[col].clip(upper=p99)

    # --- Fill ORIG AMOUNT NaN with median ---
    if "ORIG AMOUNT" in df.columns:
        df["ORIG AMOUNT"] = df["ORIG AMOUNT"].fillna(df["ORIG AMOUNT"].median())

    # --- Basic transaction aggregations ---
    billing_col = "BILLING AMT" if "BILLING AMT" in df.columns else None
    agg_dict: dict = {"txn_count": (cid, "size")}
    if billing_col:
        agg_dict.update({
            "txn_total_amount": (billing_col, "sum"),
            "txn_mean_amount":  (billing_col, "mean"),
            "txn_max_amount":   (billing_col, "max"),
        })

    agg = df.groupby(cid).agg(**agg_dict)

    # --- International transaction ratio ---
    if "CCY" in df.columns:
        df["_is_intl"] = (pd.to_numeric(df["CCY"], errors="coerce") != 818).astype(int)
        agg = agg.join(df.groupby(cid)["_is_intl"].sum().rename("txn_intl_count"))
        agg["txn_intl_ratio"] = (agg["txn_intl_count"] / agg["txn_count"]).where(
            agg["txn_count"] > 0, other=0
        )
    else:
        agg["txn_intl_count"] = 0
        agg["txn_intl_ratio"] = 0

    # --- Reversal ratio ---
    if "REVERSAL FLAG" in df.columns:
        df["_is_reversal"] = (
            df["REVERSAL FLAG"].astype(str).str.strip().str.upper()
            .isin(["Y", "YES", "1", "TRUE"])
        ).astype(int)
        agg = agg.join(df.groupby(cid)["_is_reversal"].sum().rename("txn_reversal_count"))
        agg["txn_reversal_ratio"] = (agg["txn_reversal_count"] / agg["txn_count"]).where(
            agg["txn_count"] > 0, other=0
        )
    else:
        agg["txn_reversal_count"] = 0
        agg["txn_reversal_ratio"] = 0

    # --- Merchant diversity ---
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

def create_target(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Create the target column (and optionally sample_weight) in-place.

    Returns the full DataFrame so callers can access both 'target' and
    'sample_weight' without silent column loss.

    Modes
    -----
    hard_only       : 1 for 60DA / 90DA / WROF only  (recommended first step)
    weighted_binary : 1 for all defaults; soft defaults get weight = 0.4
    tiered          : 0 = normal, 1 = soft default, 2 = hard default
    all             : 1 for all DEFAULT_STATUSES (original behaviour)
    """
    if mode is None:
        mode = getattr(config, "TARGET_MODE", "all")

    df = df.copy()

    if mode == "hard_only":
        df["target"] = df[config.STATUS_COL].isin(
            config.HARD_DEFAULT_STATUSES
        ).astype(int)

    elif mode == "weighted_binary":
        all_defaults = config.HARD_DEFAULT_STATUSES + config.SOFT_DEFAULT_STATUSES
        df["target"] = df[config.STATUS_COL].isin(all_defaults).astype(int)
        weight_map = {
            **{s: 1.0 for s in config.HARD_DEFAULT_STATUSES},
            **{s: 0.4 for s in config.SOFT_DEFAULT_STATUSES},
        }
        df["sample_weight"] = df[config.STATUS_COL].map(weight_map).fillna(0.0)

    elif mode == "tiered":
        df["target"] = np.where(
            df[config.STATUS_COL].isin(config.HARD_DEFAULT_STATUSES), 2,
            np.where(df[config.STATUS_COL].isin(config.SOFT_DEFAULT_STATUSES), 1, 0),
        )

    elif mode == "all":
        df["target"] = df[config.STATUS_COL].isin(
            config.DEFAULT_STATUSES
        ).astype(int)

    else:
        raise ValueError(
            f"Unknown TARGET_MODE '{mode}'. "
            "Choose from: hard_only, weighted_binary, tiered, all."
        )

    return df
