# ===== START OF config.py ===== 
"""
Central configuration for the Credit Risk Prediction System.
All paths, column names, hyperparameters, and constants live here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRIME_DATA_DIR = os.path.join(BASE_DIR, "prime_data")
TRANSACTION_DATA_DIR = os.path.join(BASE_DIR, "transaction_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
SCORES_PATH = os.path.join(OUTPUT_DIR, "risk_scores.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.txt")

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------
PRIME_CUSTOMER_ID = "RIM_NO"
TXN_CUSTOMER_ID = "RIMNO"
TARGET_COL = "target"
STATUS_COL = "STATUS"

# Statuses that count as default (target = 1)
DEFAULT_STATUSES = ["30DD", "90DA", "SUSP", "WROF"]
# Statuses that count as non-default (target = 0)
NON_DEFAULT_STATUSES = ["NORM", "CLSB", "CLSC"]

# Columns to drop before modelling (IDs, raw dates, text, etc.)
DROP_COLS = [
    "RIM_NO", "RIMNO", "MAPPING ACCOUNT NO.", "BRANCH_ID", "BRANCH_NAME",
    "NAME", "ORGANIZATION", "STATUES NAME", "STATUS",
    "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD",
    "CREATION_DATE", "LAST_STATEMENT_DATE", "LAST_PAYMENT_DATE",
    "CLOSURE_DATE", "DOB", "CARD_ACCOUNT_STATUS", "source_file",
]

# ---------------------------------------------------------------------------
# Date columns (for parsing)
# ---------------------------------------------------------------------------
DATE_COLS_PRIME = [
    "CREATION_DATE", "LAST_STATEMENT_DATE", "LAST_PAYMENT_DATE",
    "CLOSURE_DATE", "DOB",
]
DATE_COLS_TXN = ["POST DATE", "TRXN DATE"]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.3

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "max_depth": 4,
    "num_leaves": 55,
    "min_child_weight": 5,
    "bagging_fraction": 0.9,
    "feature_fraction": 0.9,
    "lambda_l1": 1e-5,
    "lambda_l2": 100,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
TUNE_N_ITER = 20
TUNE_CV_FOLDS = 3

TUNE_PARAM_GRID = {
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 4, 5, 6],
    "model__num_leaves": [31, 55, 80, 100],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "model__feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "model__lambda_l1": [1e-5, 1e-3, 1e-2],
    "model__lambda_l2": [1, 10, 100],
}
 
# ===== START OF data_loader.py ===== 
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
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
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
 
# ===== START OF evaluation.py ===== 
"""
Evaluation metrics and report generation.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, confusion_matrix, classification_report,
)

import config


def ks_statistic(y_true, y_proba) -> float:
    """Kolmogorov-Smirnov statistic: max |TPR - FPR|."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(y_true, y_pred, y_proba) -> dict:
    """Compute a full set of binary-classification metrics."""
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "KS": ks_statistic(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def generate_report(metrics: dict, y_true, y_pred,
                     output_path: str = None) -> str:
    """Build a plain-text evaluation report and optionally write it to disk."""
    lines = []
    lines.append("=" * 60)
    lines.append("CREDIT RISK MODEL — EVALUATION REPORT")
    lines.append("=" * 60)

    # Scalar metrics
    lines.append("")
    lines.append("METRICS")
    lines.append("-" * 40)
    for name, value in metrics.items():
        lines.append(f"  {name:<12s}: {value:.4f}")

    # Class distribution
    lines.append("")
    lines.append("CLASS DISTRIBUTION (test set)")
    lines.append("-" * 40)
    total = len(y_true)
    n_default = int(np.sum(y_true == 1))
    n_normal = total - n_default
    lines.append(f"  Non-default (0): {n_normal:>7,}  ({n_normal/total*100:.1f}%)")
    lines.append(f"  Default     (1): {n_default:>7,}  ({n_default/total*100:.1f}%)")

    # Confusion matrix
    lines.append("")
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    lines.append(f"  TN={cm[0,0]:>7,}   FP={cm[0,1]:>7,}")
    lines.append(f"  FN={cm[1,0]:>7,}   TP={cm[1,1]:>7,}")

    # Classification report
    lines.append("")
    lines.append("CLASSIFICATION REPORT")
    lines.append("-" * 40)
    lines.append(classification_report(y_true, y_pred, target_names=["Non-default", "Default"]))

    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[evaluation] Report saved to {output_path}")

    return report
 
# ===== START OF feature_engineering.py ===== 
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
 
# ===== START OF main.py ===== 
"""
CLI entry point for the Credit Risk Prediction System.

Usage
-----
    python main.py train          # Train with default LightGBM params
    python main.py tune           # Train with hyperparameter tuning
    python main.py score          # Score new data with a saved model
"""

import argparse
import sys

from pipeline import run_training_pipeline, run_scoring_pipeline
import config


def main():
    parser = argparse.ArgumentParser(
        description="Credit Risk Prediction System",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    sub.add_parser("train", help="Train the model with default hyperparameters")

    # --- tune ---
    sub.add_parser("tune", help="Train with RandomizedSearchCV hyperparameter tuning")

    # --- score ---
    score_parser = sub.add_parser("score", help="Score new data using a saved model")
    score_parser.add_argument(
        "--model", default=config.MODEL_PATH,
        help="Path to saved model (.joblib)",
    )
    score_parser.add_argument(
        "--prime-dir", default=config.PRIME_DATA_DIR,
        help="Directory with new prime CSVs",
    )
    score_parser.add_argument(
        "--txn-dir", default=config.TRANSACTION_DATA_DIR,
        help="Directory with new transaction CSVs",
    )
    score_parser.add_argument(
        "--output", default=config.SCORES_PATH,
        help="Output CSV path for risk scores",
    )

    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training_pipeline(tune=False)
        print("\nDone. Metrics:", metrics)

    elif args.command == "tune":
        metrics = run_training_pipeline(tune=True)
        print("\nDone. Metrics:", metrics)

    elif args.command == "score":
        run_scoring_pipeline(
            model_path=args.model,
            prime_dir=args.prime_dir,
            txn_dir=args.txn_dir,
            output_path=args.output,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
 
# ===== START OF model.py ===== 
"""
Model training, SMOTE resampling, hyperparameter tuning, and persistence.
"""

import numpy as np
import lightgbm as lgb
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

import config


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to the training set."""
    sm = SMOTE(sampling_strategy="auto", random_state=config.RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(
        f"[model] SMOTE: {X_train.shape[0]} -> {X_res.shape[0]} samples  "
        f"(minority: {int(y_train.sum())} -> {int(y_res.sum())})"
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train a LightGBM model with early stopping.

    Returns the trained Booster.
    """
    params = params or config.LGBM_PARAMS

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    bst = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"[model] Best iteration: {bst.best_iteration}")
    return bst


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Run RandomizedSearchCV over a SMOTE + LightGBM pipeline.

    Returns (best_estimator, best_params).
    """
    pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="binary",
                    metric="auc",
                    boosting_type="gbdt",
                    verbosity=-1,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=config.TUNE_PARAM_GRID,
        n_iter=config.TUNE_N_ITER,
        scoring=scorer,
        cv=config.TUNE_CV_FOLDS,
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print("[model] Starting hyperparameter search ...")
    search.fit(X_train, y_train)

    print(f"[model] Best CV AUC: {search.best_score_:.4f}")
    print(f"[model] Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, artifacts: dict, path: str = None):
    """Persist model and preprocessing artifacts."""
    path = path or config.MODEL_PATH
    payload = {"model": model, "artifacts": artifacts}
    joblib.dump(payload, path)
    print(f"[model] Saved to {path}")


def load_model(path: str = None):
    """Load model and preprocessing artifacts."""
    path = path or config.MODEL_PATH
    payload = joblib.load(path)
    print(f"[model] Loaded from {path}")
    return payload["model"], payload["artifacts"]
 
# ===== START OF pipeline.py ===== 
"""
Pipeline orchestrator — ties data loading, feature engineering,
preprocessing, training, and evaluation together.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import load_prime_data, load_transaction_data, merge_data
from feature_engineering import (
    engineer_prime_features,
    engineer_transaction_features,
    create_target,
)
from preprocessing import preprocess
from model import apply_smote, train_lightgbm, tune_hyperparameters, save_model, load_model
from evaluation import evaluate, generate_report


def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(tune: bool = False):
    """Full training pipeline: load -> engineer -> preprocess -> train -> evaluate.

    Parameters
    ----------
    tune : bool
        If True, run RandomizedSearchCV for hyperparameter tuning instead
        of training with the default parameters.
    """
    _ensure_output_dir()

    # 1. Load data
    prime_df = load_prime_data()
    txn_df = load_transaction_data()

    # 2. Feature engineering
    prime_df = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)

    # 3. Merge
    merged = merge_data(prime_df, txn_features)

    # 4. Create target
    target = create_target(merged)

    # Keep only rows that got a valid target
    merged = merged.loc[target.index]

    # Preserve customer IDs for the scored output
    customer_ids = merged[config.PRIME_CUSTOMER_ID].copy()

    # 5. Preprocess
    X, y, artifacts = preprocess(merged, target, fit=True)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(
        f"[pipeline] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | "
        f"Features: {X_train.shape[1]}"
    )

    # 7 + 8. Train (with optional tuning)
    if tune:
        best_model, best_params = tune_hyperparameters(X_train, y_train)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        model_to_save = best_model
    else:
        # SMOTE on training set
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        # Train LightGBM with early stopping
        bst = train_lightgbm(X_train_res, y_train_res, X_test, y_test)
        y_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
        y_pred = (y_proba >= 0.5).astype(int)
        model_to_save = bst

    # 9. Evaluate
    metrics = evaluate(y_test, y_pred, y_proba)
    report = generate_report(metrics, y_test, y_pred, config.REPORT_PATH)
    print(report)

    # 10. Score all customers & write CSV
    if tune:
        all_proba = best_model.predict_proba(X)[:, 1]
    else:
        all_proba = bst.predict(X, num_iteration=bst.best_iteration)
    all_pred = (all_proba >= 0.5).astype(int)

    scores_df = pd.DataFrame({
        config.PRIME_CUSTOMER_ID: customer_ids.loc[X.index].values,
        "default_probability": all_proba,
        "predicted_label": all_pred,
    })
    scores_df.to_csv(config.SCORES_PATH, index=False)
    print(f"[pipeline] Risk scores saved to {config.SCORES_PATH}")

    # 11. Save model + artifacts
    save_model(model_to_save, artifacts)

    return metrics


# ---------------------------------------------------------------------------
# Scoring pipeline (new data)
# ---------------------------------------------------------------------------

def run_scoring_pipeline(model_path: str = None,
                         prime_dir: str = None,
                         txn_dir: str = None,
                         output_path: str = None):
    """Score new monthly data using a previously trained model.

    Parameters
    ----------
    model_path : str
        Path to the saved model joblib file.
    prime_dir : str
        Directory containing prime CSVs for the new month.
    txn_dir : str
        Directory containing transaction CSVs for the new month.
    output_path : str
        Where to write the scored CSV.
    """
    _ensure_output_dir()
    output_path = output_path or config.SCORES_PATH

    # Load model + artifacts
    model, artifacts = load_model(model_path)

    # Load new data
    prime_df = load_prime_data(prime_dir)
    txn_df = load_transaction_data(txn_dir)

    # Feature engineering
    prime_df = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)
    merged = merge_data(prime_df, txn_features)

    customer_ids = merged[config.PRIME_CUSTOMER_ID].copy()

    # Preprocess (transform only — reuse fitted artifacts)
    X, _, _ = preprocess(merged, y=None, fit=False, artifacts=artifacts)

    # Predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X, num_iteration=model.best_iteration)

    pred = (proba >= 0.5).astype(int)

    scores_df = pd.DataFrame({
        config.PRIME_CUSTOMER_ID: customer_ids.values,
        "default_probability": proba,
        "predicted_label": pred,
    })
    scores_df.to_csv(output_path, index=False)
    print(f"[pipeline] Scored {len(scores_df):,} customers -> {output_path}")

    return scores_df
 
# ===== START OF preprocessing.py ===== 
"""
Preprocessing: missing-value imputation, encoding, and scaling.
Supports fit (training) and transform-only (scoring) modes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def preprocess(X: pd.DataFrame, y: pd.Series = None,
               fit: bool = True, artifacts: dict = None):
    """Preprocess features for modelling.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix (after feature engineering, before modelling).
    y : pd.Series, optional
        Target vector.  Only needed when ``fit=True`` to align indices.
    fit : bool
        If True, fit encoders / scaler and return them in *artifacts*.
        If False, reuse the encoders / scaler from *artifacts*.
    artifacts : dict, optional
        Previously fitted {encoders, scaler, cat_cols, num_cols, feature_order}.

    Returns
    -------
    X_out : pd.DataFrame
    y_out : pd.Series  (or None when y is None)
    artifacts : dict
    """
    X = X.copy()

    # --- Drop columns that should not be features ---
    cols_to_drop = [c for c in config.DROP_COLS if c in X.columns]
    X = X.drop(columns=cols_to_drop, errors="ignore")

    if fit:
        # Discover column types
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        # --- Missing-value imputation ---
        medians = {}
        modes = {}
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            med = X[c].median()
            medians[c] = med
            X[c] = X[c].fillna(med)
        for c in cat_cols:
            mode_val = X[c].mode().iloc[0] if not X[c].mode().empty else "UNKNOWN"
            modes[c] = mode_val
            X[c] = X[c].fillna(mode_val)

        # --- Label-encode categoricals ---
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            encoders[c] = le

        # --- Standard-scale numericals ---
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        artifacts = {
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "encoders": encoders,
            "scaler": scaler,
            "medians": medians,
            "modes": modes,
            "feature_order": X.columns.tolist(),
        }
    else:
        # --- Transform mode: reuse fitted artifacts ---
        if artifacts is None:
            raise ValueError("artifacts must be provided when fit=False")

        cat_cols = artifacts["cat_cols"]
        num_cols = artifacts["num_cols"]

        for c in num_cols:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X[c] = X[c].fillna(artifacts["medians"].get(c, 0))
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].fillna(artifacts["modes"].get(c, "UNKNOWN"))

        for c in cat_cols:
            if c in X.columns:
                le = artifacts["encoders"][c]
                # Handle unseen labels gracefully
                X[c] = X[c].astype(str).map(
                    lambda v, _le=le: (
                        _le.transform([v])[0] if v in _le.classes_ else -1
                    )
                )

        existing_num = [c for c in num_cols if c in X.columns]
        if existing_num:
            X[existing_num] = artifacts["scaler"].transform(X[existing_num])

        # Ensure same column order; add missing columns as 0
        for c in artifacts["feature_order"]:
            if c not in X.columns:
                X[c] = 0
        X = X[artifacts["feature_order"]]

    # Align indices
    if y is not None:
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]

    print(f"[preprocess] Output shape: {X.shape}  |  fit={fit}")
    return X, y, artifacts
