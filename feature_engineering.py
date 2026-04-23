"""
Feature engineering for prime (customer snapshot) and transaction data.
Preprocessing approach adapted from data_to_to_apply.py.

Includes:
- Missing-value imputation with indicator flags (DOB, LAST_STAEMENT_DATE)
- Account tenure, customer age, card replacement frequency
- Product cross-tabulation (multi-hot product ownership)
- RFM transaction features (recency, frequency, monetary)
- MCC-level spend pivot
- Foreign transaction features
- Age & credit-limit banding with one-hot encoding
- StandardScaler on float columns
- Final NaN filling (median for numeric, mode for categorical)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

# ---------------------------------------------------------------------------
# Columns that directly encode the target and must never become features.
# ---------------------------------------------------------------------------
_LEAKAGE_COLS = {
    "STATUS", "STATUS_NAME", "DELINQUENCY",
    "CARD_ACCOUNT_STATUS", "Card account status ",
    "CLOSURE_DATE",          # non-null only for defaulted/closed accounts
}


# ---------------------------------------------------------------------------
# Prime features
# ---------------------------------------------------------------------------

def engineer_prime_features(
    prime_df: pd.DataFrame,
    txn_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive all features from the prime snapshot *and* transaction data.

    This replaces both the old ``engineer_prime_features`` and
    ``engineer_transaction_features`` — all enrichment happens in one place
    so we can merge transaction-derived signals (last-statement fallback,
    RFM, MCC spend, foreign-txn ratio, etc.) directly.

    Parameters
    ----------
    prime_df : pd.DataFrame
        Raw prime data (after loading / casting / column-dropping).
    txn_df : pd.DataFrame
        Raw transaction data (after loading / casting / column-dropping).

    Returns
    -------
    pd.DataFrame
        Enriched prime DataFrame ready for target creation + preprocessing.
    """
    df = prime_df.copy()
    extraction_date = pd.to_datetime("today")
    cid = config.CUSTOMER_ID

    # ── 1. Fill known missing-value patterns ──────────────────────────
    df["GENDER"] = df["GENDER"].fillna("Unknown")

    # DOB: flag + median impute
    df["DOB_IS_MISSING"] = df["DOB"].isna().astype(int)
    if "DOB" in df.columns and not df["DOB"].dropna().empty:
        median_dob_int = df["DOB"].dropna().astype("int64").median()
        median_dob = pd.to_datetime(median_dob_int)
        df["DOB"] = df["DOB"].fillna(median_dob)

    # LAST_STAEMENT_DATE: flag + fill from last txn date, then CREATION_DATE
    df["LAST_STAEMENT_IS_MISSING"] = df["LAST_STAEMENT_DATE"].isna().astype(int) \
        if "LAST_STAEMENT_DATE" in df.columns else 0
    if "LAST_STAEMENT_DATE" in df.columns:
        latest_txns = txn_df.groupby(cid)["TRXN DATE"].max()
        df["LAST_STAEMENT_DATE"] = df["LAST_STAEMENT_DATE"].fillna(
            df[cid].map(latest_txns)
        )
        df["LAST_STAEMENT_DATE"] = df["LAST_STAEMENT_DATE"].fillna(
            df["CREATION_DATE"]
        )

    # Transaction categorical fills
    txn_df = txn_df.copy()
    txn_df["MERCHNAME"]     = txn_df["MERCHNAME"].fillna("UNKNOWN")
    txn_df["BANKBRANCH"]    = txn_df["BANKBRANCH"].fillna("UNKNOWN")
    txn_df["TRXN COUNTRY"]  = txn_df["TRXN COUNTRY"].fillna("UNKNOWN")

    # ── 2. Temporal features ──────────────────────────────────────────
    if "CREATION_DATE" in df.columns:
        df["ACCOUNT_TENURE_MONTHS"] = (
            (extraction_date - df["CREATION_DATE"]).dt.days / 30
        )

    # NOTE: TIME_TO_CHURN_DAYS intentionally omitted — it relies on
    # CLOSURE_DATE which is a leakage source (non-null only for
    # closed/defaulted accounts) and is therefore incorrect.

    # ── 3. Days since last transaction (recency from txn data) ────────
    txn_recency = txn_df.groupby(cid)["TRXN DATE"].max().reset_index()
    txn_recency["DAYS_SINCE_LAST_TXN"] = (
        extraction_date - txn_recency["TRXN DATE"]
    ).dt.days
    df = df.merge(
        txn_recency[[cid, "DAYS_SINCE_LAST_TXN"]],
        on=cid, how="left",
    )

    # ── 4. Card replacement frequency ────────────────────────────────
    rep_cols = ["FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD"]
    existing_rep = [c for c in rep_cols if c in df.columns]
    if existing_rep:
        df["CARD_REPLACEMENT_FREQ"] = df[existing_rep].sum(axis=1)

    # ── 5. Product cross-tabulation ──────────────────────────────────
    if "PRODUCT_NAME" in df.columns:
        Product_frequency_threshold = 500
        prime_counts = df["PRODUCT_NAME"].value_counts()
        df["PRODUCT_NAME"] = df["PRODUCT_NAME"].map(
            lambda x: x if prime_counts.get(x, 0) >= Product_frequency_threshold
            else "another product"
        )

        user_item_matrix = pd.crosstab(df[cid], df["PRODUCT_NAME"])
        user_item_matrix.columns = [
            f"HAS_PROD_{col.strip().replace(' ', '_')}"
            for col in user_item_matrix.columns
        ]
        df = (
            df.drop_duplicates(subset=[cid])
              .merge(user_item_matrix, on=cid, how="left")
        )

    # Also threshold PRODUCT_NAME in transactions
    if "PRODUCT_NAME" in txn_df.columns:
        txn_prod_counts = txn_df["PRODUCT_NAME"].value_counts()
        txn_df["PRODUCT_NAME"] = txn_df["PRODUCT_NAME"].map(
            lambda x: x if txn_prod_counts.get(x, 0) >= 500
            else "another product"
        )

    # ── 6. RFM features (Recency, Frequency, Monetary) ───────────────
    rfm_features = txn_df.groupby(cid).agg(
        TOTAL_SPEND_AMT=("BILLING AMT", "sum"),
        AVG_TRXN_AMT=("BILLING AMT", "mean"),
        TRXN_COUNT=("BILLING AMT", "count"),
        DAYS_SINCE_LAST_TRXN=(
            "TRXN DATE",
            lambda x: (pd.to_datetime("today") - x.max()).days,
        ),
    ).reset_index()

    # ── 7. MCC-level spend pivot ─────────────────────────────────────
    MCC_frequency_threshold = 5000
    if "MCC" in txn_df.columns:
        mcc_counts = txn_df["MCC"].value_counts()
        txn_df["MCC"] = txn_df["MCC"].map(
            lambda x: x if mcc_counts.get(x, 0) >= MCC_frequency_threshold
            else "other_mcc"
        )
        mcc_spend = pd.pivot_table(
            txn_df,
            values="BILLING AMT",
            index=cid,
            columns="MCC",
            aggfunc="sum",
            fill_value=0,
        )
        mcc_spend.columns = [f"MCC_{col}_SPEND" for col in mcc_spend.columns]
        mcc_spend = mcc_spend.reset_index()
    else:
        mcc_spend = pd.DataFrame({cid: txn_df[cid].unique()})

    # ── 8. Foreign transaction features ──────────────────────────────
    if "TRXN COUNTRY" in txn_df.columns:
        txn_df["IS_FOREIGN_TRXN"] = (
            txn_df["TRXN COUNTRY"] != "EG"
        ).fillna(False).astype(int)
        travel_features = txn_df.groupby(cid).agg(
            FOREIGN_TRXN_COUNT=("IS_FOREIGN_TRXN", "sum"),
            FOREIGN_SPEND_RATIO=("IS_FOREIGN_TRXN", "mean"),
        ).reset_index()
    else:
        travel_features = pd.DataFrame({cid: txn_df[cid].unique()})
        travel_features["FOREIGN_TRXN_COUNT"] = 0
        travel_features["FOREIGN_SPEND_RATIO"] = 0.0

    # ── 9. Merge transaction features back ────────────────────────────
    df = df.merge(rfm_features, on=cid, how="left")
    df = df.merge(mcc_spend, on=cid, how="left")
    df = df.merge(travel_features, on=cid, how="left")

    # Fill NaN for customers with no transactions in these new columns
    trxn_feature_cols = (
        [c for c in rfm_features.columns if c != cid]
        + [c for c in mcc_spend.columns if c != cid]
        + [c for c in travel_features.columns if c != cid]
    )
    for c in trxn_feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ── 10. Customer age & bands ─────────────────────────────────────
    if "DOB" in df.columns:
        df["AGE"] = (extraction_date - df["DOB"]).dt.days // 365
        bins   = [18, 25, 35, 50, 65, 100]
        labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        df["AGE_GROUP"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=True)

    # ── 11. Credit-limit bands ───────────────────────────────────────
    if "CREDIT_LIMIT" in df.columns:
        limit_bins   = [0, 10_000, 50_000, 100_000, 500_000, np.inf]
        limit_labels = ["Low", "Medium", "High", "Very High", "Premium"]
        df["LIMIT_BAND"] = pd.cut(
            df["CREDIT_LIMIT"], bins=limit_bins, labels=limit_labels,
        )

    # ── 12. One-hot encoding ─────────────────────────────────────────
    cols_to_ohe = ["GENDER", "AGE_GROUP", "LIMIT_BAND", "CUSTOMER_TYPE", "ACTIVATED"]
    existing_ohe = [c for c in cols_to_ohe if c in df.columns]
    if existing_ohe:
        df = pd.get_dummies(df, columns=existing_ohe, drop_first=True)

    # ── 13. StandardScaler on float columns ──────────────────────────
    print("\n--- Applying Data Scaling ---")
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    if float_cols:
        scaler = StandardScaler()
        print(f"Scaling {len(float_cols)} float columns ...")
        df[float_cols] = scaler.fit_transform(df[float_cols])
    else:
        print("No float columns found to scale.")

    # ── 14. Final NaN sweep ──────────────────────────────────────────
    print("\n--- Final Check for NaN Values ---")
    cols_with_nans = df.columns[df.isna().any()].tolist()
    if not cols_with_nans:
        print("Clean sweep! No NaN values found.")
    else:
        print(
            f"Found {len(cols_with_nans)} columns with missing values. "
            "Filling with median / mode ..."
        )
        for col in cols_with_nans:
            if pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f" - Filled [{col}] with median: {med:.4f}")
            else:
                mode_val = (
                    df[col].mode()[0]
                    if not df[col].mode().empty
                    else "Unknown"
                )
                df[col] = df[col].fillna(mode_val)
                print(f" - Filled non-numeric [{col}] with mode: {mode_val}")

    print(f"Final NaN count: {df.isna().sum().sum()}")
    print(f"[feature_eng] Prime features engineered -> {df.shape}")
    return df


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
