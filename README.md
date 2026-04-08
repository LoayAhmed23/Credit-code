# Credit Risk Prediction System

Binary credit-card default prediction system built on LightGBM. Ingests monthly customer snapshots and transaction records, engineers features, trains a classifier, and outputs per-customer risk scores with an evaluation report.

## Project Structure

```
Credit/
├── config.py                # Paths, column names, hyperparameters, constants
├── data_loader.py           # Read & merge monthly CSVs from both data folders
├── feature_engineering.py   # Derive features from prime + transaction data
├── preprocessing.py         # Encode, scale, impute missing values
├── model.py                 # LightGBM training, SMOTE, tuning, persistence
├── evaluation.py            # Metrics (AUC, KS, F1, confusion matrix), reports
├── pipeline.py              # Orchestrates the full end-to-end flow
├── main.py                  # CLI entry point
├── prime_data/              # Monthly customer-snapshot CSVs (user-provided)
├── transaction_data/        # Monthly transaction CSVs (user-provided)
└── output/                  # Generated model, scores, and reports
```

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
imbalanced-learn
joblib
```

Install with:

```bash
pip install pandas numpy scikit-learn lightgbm imbalanced-learn joblib
```

## Data Setup

### 1. Prime Data (`prime_data/`)

Place monthly customer-snapshot CSVs here. Each file should contain one row per credit-card account with these columns:

| Column | Description |
|--------|-------------|
| RIM_NO | Customer ID |
| MAPPING ACCOUNT NO. | Account number |
| BRANCH_ID / BRANCH_NAME | Branch info |
| CREATION_DATE | Account creation date |
| CREDIT LIMIT | Card credit limit |
| ACTIVATED | A (active) / I (inactive) |
| STATUS | 4-letter code: NORM, CLSB, CLSC, WROF, 90DA, 30DD, SUSP |
| STATUES NAME | Description matching STATUS |
| DELINQUENCY | Days delinquent |
| LAST_STATEMENT_DATE | Date of last statement |
| LEDGER_BALANCE | Current balance |
| MIN_PAYMENT | Minimum payment due |
| OVER_LIMIT | Amount over credit limit |
| AVAILABLE_LIMIT | Remaining credit limit |
| LAST_PAYMENT_AMOUNT | Last payment amount |
| LAST_PAYMENT_DATE | Date of last payment |
| GENDER / DOB | Demographics |
| ORGANIZATION | Employer |
| CUSTOMER_TYPE | Primary / secondary card |
| OVERDUE_AMOUNT | Overdue amount |
| NO_OF_CYCLES | Billing cycles since opening |
| JOINING_FEE / ANNUAL_FEE | Fee info |
| FIRST/SECOND/THIRD_REPLACED_CARD | Replacement card flags |
| CARD_ACCOUNT_STATUS | Account status |

### 2. Transaction Data (`transaction_data/`)

Place monthly transaction CSVs here. Each file should contain individual transactions with these columns:

| Column | Description |
|--------|-------------|
| RIMNO | Customer ID (maps to RIM_NO in prime data) |
| DESCRIPTION | Bank product used |
| POST DATE / TRXN DATE | Posting and transaction dates |
| CCY | ISO currency code (818 = EGP, 840 = USD) |
| ORIG AMOUNT | Original transaction amount |
| BILLING AMT | Amount in EGP |
| MCC | Merchant category code |
| MERCHNAME / MERCH ID | Merchant info |
| REVERSAL FLAG | Whether the transaction was reversed |
| PaymentBehaviourHistory | Last 10 T/R characters (T = successful, R = reversed) |

## Usage

### Train with default parameters

```bash
python main.py train
```

Trains a LightGBM model with SMOTE resampling and early stopping. Uses the hyperparameters defined in `config.py`.

### Train with hyperparameter tuning

```bash
python main.py tune
```

Runs `RandomizedSearchCV` over a SMOTE + LightGBM pipeline (20 iterations, 3-fold CV by default). Searches over learning rate, max depth, num leaves, regularization, and bagging/feature fractions.

### Score new data

```bash
python main.py score
```

Loads a previously saved model and scores new monthly data. Optional arguments:

```
--model       Path to saved model (.joblib)         [default: output/model.joblib]
--prime-dir   Directory with new prime CSVs          [default: prime_data/]
--txn-dir     Directory with new transaction CSVs    [default: transaction_data/]
--output      Output CSV path                        [default: output/risk_scores.csv]
```

## Output

After training, the `output/` folder contains:

| File | Contents |
|------|----------|
| `risk_scores.csv` | RIM_NO, default_probability, predicted_label |
| `evaluation_report.txt` | AUC, KS, F1, precision, recall, accuracy, confusion matrix, classification report |
| `model.joblib` | Serialized model + preprocessing artifacts |

## Target Variable

Binary classification:

- **0 (Non-default)**: STATUS in {NORM, CLSB, CLSC}
- **1 (Default)**: STATUS in {30DD, 90DA, SUSP, WROF}

## Engineered Features

### From Prime Data
- `utilization_ratio` -- (credit limit - available limit) / credit limit
- `payment_to_min_ratio` -- last payment / min payment (capped at 10)
- `overdue_severity` -- overdue amount / credit limit
- `overlimit_flag` -- 1 if over limit > 0
- `card_replacements` -- count of replacement cards issued
- `account_age_days` -- days between creation and last statement
- `days_since_last_payment` -- days between last payment and last statement
- `is_active` -- 1 if ACTIVATED = "A"
- `customer_age` -- age derived from DOB
- `is_primary` -- 1 if primary card holder
- `ledger_to_limit_ratio` -- ledger balance / credit limit

### From Transaction Data
- `txn_count` -- total transactions
- `txn_total_amount` -- sum of billing amounts
- `txn_mean_amount` / `txn_max_amount` -- average and max transaction
- `txn_intl_count` / `txn_intl_ratio` -- international transaction count and ratio
- `txn_reversal_count` / `txn_reversal_ratio` -- reversed transaction count and ratio
- `unique_merchants` / `unique_mcc` -- merchant and category diversity
- `payment_success_rate` -- ratio of "T" characters in PaymentBehaviourHistory

## Pipeline Flow

```
prime_data/*.csv --> load --> engineer features --+
                                                  |--> merge --> create target
transaction_data/*.csv --> load --> aggregate -----+        |
                                                           v
                                                     preprocess
                                                           |
                                                   train/test split
                                                           |
                                                   SMOTE (train only)
                                                           |
                                                   LightGBM training
                                                           |
                                                   evaluate + report
                                                           |
                                              risk_scores.csv + report.txt
```

## Configuration

All settings are centralized in `config.py`:

- **Paths**: data directories, output directory, model path
- **Column mappings**: customer ID columns, status codes
- **Model params**: LightGBM hyperparameters, SMOTE settings
- **Tuning**: search space, number of iterations, CV folds

Modify `config.py` to adjust any defaults without touching the pipeline code.
