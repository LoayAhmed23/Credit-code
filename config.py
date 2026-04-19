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
CUSTOMER_ID = "RIMNO"
TARGET_COL = "target"
STATUS_COL = "STATUS"

TARGET_MODE = "weighted_binary"

SOFT_DEFAULT_STATUSES = ["30DD", "SUSP"]           # early warning — often self-corrects
HARD_DEFAULT_STATUSES = ["60DA", "90DA", "WROF"]   # true default — rarely recovers

# Keep for backward compatibility when you need a single binary label
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
PRIME_INT_COLS = ["RIMNO"]
PRIME_FLOAT_COLS = [
    "AVAILABLE_LIMIT", "LEDGER_BALANCE", "LAST_PAYMENT_AMOUNT", "OVERDUEAMOUNT", "CREDIT_LIMIT"
]

# --- Transaction string cols (read as string to avoid parse errors) ---
TXN_STRING_COLS = [
    "DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES",
    "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG",
]

# Columns to drop before modelling (IDs, raw dates, text, etc.)
DROP_COLS = [
    "RIMNO", "BRANCH_ID", "BRANCH_NAME",
    "NAME", "ORGANIZATION", "STATUS_NAME", "STATUS",
    "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD",
    "CREATION_DATE", "LAST_STAEMENT_DATE", "LAST_PAYMENT_DATE",
    "CLOSURE_DATE", "DOB", "Card account status ", "source_file",
    "OVER_LIMIT", "OVERDUEAMOUNT", "LEDGER_BALANCE", "MIN_PAYMENT_AMOUNT",
]

# ---------------------------------------------------------------------------
# Date columns (for parsing)
# ---------------------------------------------------------------------------
DATE_COLS_PRIME = [
    "CREATION_DATE", "LAST_STAEMENT_DATE", "LAST_PAYMENT_DATE",
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
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,            # Uses all available CPU cores
    "max_bin": 63,           # <--- ADDED: Significantly faster training (default is 255)
    "device_type": "cpu",    # CPU is highly optimized and avoids the best_split_info_count bug
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
TUNE_N_ITER = 20
TUNE_CV_FOLDS = 3

TUNE_PARAM_GRID = {
    "classifier__num_leaves": [15, 31, 63],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__feature_fraction": [0.6, 0.8, 1.0],
    "classifier__bagging_fraction": [0.6, 0.8, 1.0],
    "classifier__device_type": ["cpu"],  # <--- CHANGED: Ensure tuning uses CPU too
}
