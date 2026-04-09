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

# Statuses that count as default (target = 1)
DEFAULT_STATUSES = ["30DD", "60DA", "90DA", "SUSP", "WROF"]
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
