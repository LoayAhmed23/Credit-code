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
CUSTOMER_ID = "CUSTOMER_ID"
TARGET_COL = "target"
STATUS_COL = "STATUS"
MONTH_COL = "snapshot_month"

TARGET_MODE = "weighted_binary"

SOFT_DEFAULT_STATUSES = ["30DD", "SUSP"]           # early warning — often self-corrects
HARD_DEFAULT_STATUSES = ["60DA", "90DA", "WROF"]   # true default — rarely recovers

# Backward-compatible combined list
DEFAULT_STATUSES = SOFT_DEFAULT_STATUSES + HARD_DEFAULT_STATUSES

# Everything else is non-default (target = 0)
NON_DEFAULT_STATUSES = [
    "NORM", "NEW", "CLSB", "CLSC", "CLSD",
    "LOST", "CNCD", "FRAD", "BLOK", "NOAU", "EXMU",
    "PICK", "BLCK", "STLC", "OFBL", "ONBL", "WARN",
    "NENC", "FREZ", "EXPD", "ACCA", "EXPC",
]

# --- Prime CSV dtype handling (data arrives with commas in numbers) ---
PRIME_STRING_COLS = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME", "NAME",
    "GENDER", "CUSTOMER_TYPE", "Card account status ", "ORGANIZATION",
]
PRIME_INT_COLS = ["RIMNO", "CUSTOMER_ID", "BRANCH_ID"]
PRIME_FLOAT_COLS = [
    "AVAILABLE_LIMIT", "LEDGER_BALANCE", "LAST_PAYMENT_AMOUNT",
    "OVERDUEAMOUNT", "CREDIT_LIMIT",
]

# --- Transaction string cols (read as string to avoid parse errors) ---
TXN_STRING_COLS = [
    "DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES",
    "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG",
]

# ---------------------------------------------------------------------------
# DROP_COLS — columns removed before the model ever sees the data.
#
# Three leakage categories:
#   A) Direct label encodings  — these columns ARE the target in disguise
#   B) Temporal leakage        — dates that only exist for defaulted accounts
#                                (e.g. CLOSURE_DATE is null for active accounts)
#   C) Raw amounts             — replaced by ratio features; keeping both
#                                double-counts the signal and leaks the
#                                credit limit denominator back into the model
# ---------------------------------------------------------------------------
DROP_COLS = [
    # --- Identifiers (no predictive value) ---
    "RIMNO",
    "CUSTOMER_ID",
    "snapshot_month",
    "BRANCH_ID",
    "BRANCH_NAME",
    "ACTIVATED",

    # --- (A) Direct label encodings ---
    # STATUS / STATUS_NAME contain the exact strings used to build the target.
    # DELINQUENCY is the numeric form of the same information (90DA -> 90 days).
    # CARD_ACCOUNT_STATUS is another representation of the same state.
    # Any one of these alone gives the model a perfect AUC of 1.0.
    "STATUS",
    "STATUS_NAME",
    "DELINQUENCY",
    "CARD_ACCOUNT_STATUS",
    "Card account status ",      # alternate casing/spacing from raw CSV

    # --- (B) Temporal leakage ---
    # CLOSURE_DATE is non-null only for closed/defaulted accounts.
    # LAST_STATEMENT_DATE clusters near the default date for defaulted accounts.
    "CLOSURE_DATE",
    "CREATION_DATE",
    "LAST_STAEMENT_DATE",        # typo preserved from raw data
    "LAST_STATEMENT_DATE",       # clean spelling also dropped
    "LAST_PAYMENT_DATE",
    "DOB",

    # --- Free text / high-cardinality categoricals ---
    "PRODUCT_NAME",
    "ORGANIZATION",
    "DESCRIPTION",
    "MERCHNAME",
    "source_file",

    # --- Card replacement history (administrative, not behavioural) ---
    "FIRST_REPLACED_CARD",
    "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD",

    # --- (C) Raw amount columns replaced by engineered ratio features ---
    "OVER_LIMIT",
    "OVERDUEAMOUNT",
    "OVERDUE_AMOUNT",
    "LEDGER_BALANCE",
    "MIN_PAYMENT",
    "MIN_PAYMENT_AMOUNT",

    # --- Pipeline artefact ---
    "sample_weight",
    "target",
    "CUSTOMER_TYPE",
]

# ---------------------------------------------------------------------------
# Leakage detection — single-feature AUC above this triggers a hard stop.
# ---------------------------------------------------------------------------
LEAKAGE_AUC_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Correlation-based feature filtering — drop features whose absolute Pearson
# correlation with the target falls below this threshold.
# Set to 0.0 to disable (keep all features).
# ---------------------------------------------------------------------------
CORR_THRESHOLD = 0.02

# ---------------------------------------------------------------------------
# Date columns (for parsing)
# ---------------------------------------------------------------------------
DATE_COLS_PRIME = [
    "CREATION_DATE", "LAST_STAEMENT_DATE", "CLOSURE_DATE", "DOB",
]
DATE_COLS_TXN = ["POST DATE", "TRXN DATE"]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Default XGBoost params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 5,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "seed": RANDOM_STATE,
    "max_bin": 64,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# SMOTE (class imbalance handling)
# ---------------------------------------------------------------------------
SMOTE_ENABLED = True
SMOTE_SAMPLING_STRATEGY = 0.5   # ratio of minority to majority after resampling
                                # 0.5 means minority becomes 50% of majority count

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
TUNE_N_ITER = 60          # increased from 20 for better search coverage
TUNE_CV_FOLDS = 5         # increased from 3 for more robust CV estimates
N_GPUS = 2   # number of GPUs for parallel CV folds in tuning

# F-beta value for threshold tuning (lower = more precision, higher = more recall)
# 1.0 = balanced F1, 2.0 = recall-heavy, 1.5 = mild recall preference
FBETA_VALUE = 1.5

# XGBoost hyperparameters for tuning — expanded search space
from scipy.stats import uniform, randint, loguniform

TUNE_PARAM_GRID = {
    # Tree structure
    "max_depth":          randint(3, 10),          # was [3, 5, 7]
    "min_child_weight":   randint(1, 20),           # NEW: controls overfitting
    "gamma":              uniform(0, 5),            # NEW: min split loss

    # Learning rate & boosting rounds
    "learning_rate":      loguniform(0.005, 0.3),   # was [0.01, 0.05, 0.1]
    "n_estimators":       randint(100, 1500),        # NEW: number of trees

    # Sampling / regularization
    "colsample_bytree":   uniform(0.3, 0.7),        # was [0.6, 0.8, 1.0]
    "colsample_bylevel":  uniform(0.3, 0.7),        # NEW
    "subsample":          uniform(0.5, 0.5),         # was [0.6, 0.8, 1.0]

    # L1 / L2 regularization
    "reg_alpha":          loguniform(1e-3, 10),      # NEW: L1 penalty
    "reg_lambda":         loguniform(1e-3, 10),      # NEW: L2 penalty

    # Max bins for histogram
    "max_bin":            [64, 128, 256],
}

# ---------------------------------------------------------------------------
# Temporal feature engineering
# ---------------------------------------------------------------------------
TREND_COLS = [
    "utilization_ratio", "overdue_ratio",
    "financial_stress_score", "available_credit_ratio", "over_limit_ratio",
]
TXN_TREND_COLS = [
    "txn_count", "txn_total_amount", "txn_mean_amount",
]
