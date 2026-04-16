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
from evaluation import evaluate, generate_report, find_best_threshold_fbeta


def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _banner(step, total, title):
    print()
    print("=" * 60)
    print(f"  STEP {step}/{total}: {title}")
    print("=" * 60)


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
    TOTAL = 8
    _ensure_output_dir()

    # ------------------------------------------------------------------
    _banner(1, TOTAL, "LOADING DATA")
    # ------------------------------------------------------------------
    prime_df = load_prime_data()
    txn_df = load_transaction_data()

    # ------------------------------------------------------------------
    _banner(2, TOTAL, "FEATURE ENGINEERING")
    # ------------------------------------------------------------------
    prime_df = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)

    # ------------------------------------------------------------------
    _banner(3, TOTAL, "MERGING PRIME + TRANSACTION DATA")
    # ------------------------------------------------------------------
    merged = merge_data(prime_df, txn_features)

    # ------------------------------------------------------------------
    _banner(4, TOTAL, "CREATING TARGET VARIABLE")
    # ------------------------------------------------------------------
    target = create_target(merged)
    merged = merged.loc[target.index]

    n_default = int(target.sum())
    n_total = len(target)
    print(f"  Total samples:   {n_total:,}")
    print(f"  Default     (1): {n_default:,}  ({n_default / n_total * 100:.1f}%)")
    print(f"  Non-default (0): {n_total - n_default:,}  "
          f"({(n_total - n_default) / n_total * 100:.1f}%)")

    customer_ids = merged[config.CUSTOMER_ID].copy()

    # ------------------------------------------------------------------
    _banner(5, TOTAL, "PREPROCESSING (encoding, scaling, imputation)")
    # ------------------------------------------------------------------
    X, y, artifacts = preprocess(merged, target, fit=True)

    # ------------------------------------------------------------------
    _banner(6, TOTAL, "TRAIN / TEST SPLIT")
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(f"  Train set:          {X_train.shape[0]:,} samples")
    print(f"  Test set:           {X_test.shape[0]:,} samples")
    print(f"  Features:           {X_train.shape[1]}")
    print(f"  Train default rate: {y_train.sum() / len(y_train) * 100:.1f}%")
    print(f"  Test  default rate: {y_test.sum() / len(y_test) * 100:.1f}%")

    # ------------------------------------------------------------------
    if tune:
        _banner(7, TOTAL, "HYPERPARAMETER TUNING + TRAINING")
    else:
        _banner(7, TOTAL, "SMOTE RESAMPLING + LIGHTGBM TRAINING")
    # ------------------------------------------------------------------

    if tune:
        best_model, best_params = tune_hyperparameters(X_train, y_train)
        print("  Generating predictions on test set ...")
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        print("  Finding best decision threshold (F-beta) ...")
        best_threshold = find_best_threshold_fbeta(y_test, y_proba, beta=2.0)
        print(f"  Best threshold: {best_threshold:.4f}")
        
        y_pred = (y_proba >= best_threshold).astype(int)
        model_to_save = best_model
    else:
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        print("  Training LightGBM with early stopping ...")
        bst = train_lightgbm(X_train_res, y_train_res, X_test, y_test)
        print("  Generating predictions on test set ...")
        y_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
        
        print("  Finding best decision threshold (F-beta) ...")
        best_threshold = find_best_threshold_fbeta(y_test, y_proba, beta=2.0)
        print(f"  Best threshold: {best_threshold:.4f}")
        
        y_pred = (y_proba >= best_threshold).astype(int)
        model_to_save = bst

    # ------------------------------------------------------------------
    _banner(8, TOTAL, "EVALUATION & OUTPUT")
    # ------------------------------------------------------------------
    metrics = evaluate(y_test, y_pred, y_proba, threshold=best_threshold)
    report = generate_report(metrics, y_test, y_pred, config.REPORT_PATH)
    print(report)

    print("  Scoring all customers ...")
    if tune:
        all_proba = best_model.predict_proba(X)[:, 1]
    else:
        all_proba = bst.predict(X, num_iteration=bst.best_iteration)
    all_pred = (all_proba >= best_threshold).astype(int)

    scores_df = pd.DataFrame({
        config.CUSTOMER_ID: customer_ids.loc[X.index].values,
        "default_probability": all_proba,
        "predicted_label": all_pred,
    })
    scores_df.to_csv(config.SCORES_PATH, index=False)
    print(f"  Risk scores saved to {config.SCORES_PATH}")

    save_model(model_to_save, artifacts)

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# Scoring pipeline (new data)
# ---------------------------------------------------------------------------

def run_scoring_pipeline(model_path: str = None,
                         prime_dir: str = None,
                         txn_dir: str = None,
                         output_path: str = None):
    """Score new monthly data using a previously trained model."""
    TOTAL = 5
    _ensure_output_dir()
    output_path = output_path or config.SCORES_PATH

    _banner(1, TOTAL, "LOADING SAVED MODEL")
    model, artifacts = load_model(model_path)

    _banner(2, TOTAL, "LOADING NEW DATA")
    prime_df = load_prime_data(prime_dir)
    txn_df = load_transaction_data(txn_dir)

    _banner(3, TOTAL, "FEATURE ENGINEERING")
    prime_df = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)
    merged = merge_data(prime_df, txn_features)

    customer_ids = merged[config.CUSTOMER_ID].copy()

    _banner(4, TOTAL, "PREPROCESSING")
    X, _, _ = preprocess(merged, y=None, fit=False, artifacts=artifacts)

    _banner(5, TOTAL, "SCORING")
    print(f"  Scoring {len(X):,} card accounts ...")
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
    print(f"  Saved to {output_path}")

    n_flagged = int(pred.sum())
    print(f"  Flagged as default: {n_flagged:,} / {len(pred):,} "
          f"({n_flagged / len(pred) * 100:.1f}%)")

    print()
    print("=" * 60)
    print("  SCORING COMPLETE")
    print("=" * 60)

    return scores_df
