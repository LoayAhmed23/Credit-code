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
PRIME_INT_COLS = ["RIMNO"]
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
    "BRANCH_ID",
    "BRANCH_NAME",
    "MAPPING ACCOUNT NO.",

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
    "NAME",
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
]

# ---------------------------------------------------------------------------
# Leakage detection — single-feature AUC above this triggers a hard stop.
# ---------------------------------------------------------------------------
LEAKAGE_AUC_THRESHOLD = 0.95

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

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 5,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda",
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
TUNE_N_ITER = 20
TUNE_CV_FOLDS = 3

# IMPORTANT: prefix must match the pipeline step name "model", not "classifier"
TUNE_PARAM_GRID = {
    "model__max_depth":        [3, 5, 7],
    "model__learning_rate":    [0.01, 0.05, 0.1],    
    "model__colsample_bytree": [0.6, 0.8, 1.0],      
    "model__subsample":        [0.6, 0.8, 1.0],      
    "model__tree_method":      ["hist"],
    "model__device":           ["cuda"],
}
