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
    customer_ids = merged[config.CUSTOMER_ID].copy()

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
        config.CUSTOMER_ID: customer_ids.loc[X.index].values,
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

    customer_ids = merged[config.CUSTOMER_ID].copy()

    # Preprocess (transform only — reuse fitted artifacts)
    X, _, _ = preprocess(merged, y=None, fit=False, artifacts=artifacts)

    # Predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X, num_iteration=model.best_iteration)

    pred = (proba >= 0.5).astype(int)

    scores_df = pd.DataFrame({
        config.CUSTOMER_ID: customer_ids.values,
        "default_probability": proba,
        "predicted_label": pred,
    })
    scores_df.to_csv(output_path, index=False)
    print(f"[pipeline] Scored {len(scores_df):,} customers -> {output_path}")

    return scores_df
