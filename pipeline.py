"""
Pipeline orchestrator — ties data loading, feature engineering,
preprocessing, training, and evaluation together.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

import config
from data_loader import load_prime_data, load_transaction_data, merge_data
from feature_engineering import (
    engineer_prime_features,
    engineer_transaction_features,
    create_target,
)
from preprocessing import preprocess
from model import apply_smote, train_xgboost, tune_hyperparameters, save_model, load_model
from evaluation import evaluate, generate_report, find_best_threshold_fbeta
from feature_importance import plot_feature_importance

def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _banner(step, total, title):
    print()
    print("=" * 60)
    print(f"  STEP {step}/{total}: {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def check_for_leakage(X: pd.DataFrame, y: pd.Series,
                      threshold: float = None,
                      top_n: int = 10,
                      abort_on_leak: bool = True) -> list[tuple[str, float]]:
    """Fit a depth-1 decision tree on each feature individually.

    Any feature with a single-feature AUC above `threshold` is flagged as a
    leakage suspect — it alone is almost perfectly predictive of the target,
    which should not be possible with legitimate pre-event features.

    Parameters
    ----------
    X               : feature matrix (post-preprocessing)
    y               : binary target series
    threshold       : AUC cutoff (defaults to config.LEAKAGE_AUC_THRESHOLD)
    top_n           : how many suspects to print
    abort_on_leak   : if True, raise RuntimeError when suspects are found
                      so the pipeline halts before wasting training time

    Returns
    -------
    List of (column_name, auc) tuples sorted descending by AUC.
    """
    threshold = threshold or getattr(config, "LEAKAGE_AUC_THRESHOLD", 0.95)
    suspects = []

    for col in X.columns:
        col_vals = X[[col]].fillna(-999)
        try:
            clf = DecisionTreeClassifier(max_depth=1, random_state=config.RANDOM_STATE)
            clf.fit(col_vals, y)
            auc = roc_auc_score(y, clf.predict_proba(col_vals)[:, 1])
            if auc > threshold:
                suspects.append((col, round(auc, 4)))
        except Exception:
            continue

    suspects.sort(key=lambda x: -x[1])

    if suspects:
        print(f"\n{'!' * 60}")
        print(f"  LEAKAGE WARNING — {len(suspects)} feature(s) with single-feature "
              f"AUC > {threshold}:")
        for col, auc in suspects[:top_n]:
            print(f"    {col:<45} AUC = {auc}")
        print(f"  Add these columns to config.DROP_COLS and re-run.")
        print(f"{'!' * 60}\n")

        if abort_on_leak:
            raise RuntimeError(
                f"Pipeline aborted: {len(suspects)} leakage suspect(s) found. "
                f"See output above. Set abort_on_leak=False to override."
            )
    else:
        print(f"  [leakage check] No suspects found above AUC {threshold}. Proceeding.")

    return suspects


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(tune: bool = False, sample: bool = False):
    """Full training pipeline: load -> engineer -> preprocess -> train -> evaluate.

    Parameters
    ----------
    tune : bool
        If True, run RandomizedSearchCV for hyperparameter tuning instead
        of training with the default parameters.
    sample : bool
        If True, use only 25% of the data (stratified) for fast iteration.
    """
    TOTAL = 9
    _ensure_output_dir()

    # ------------------------------------------------------------------
    _banner(1, TOTAL, "LOADING DATA")
    # ------------------------------------------------------------------
    prime_df = load_prime_data()
    txn_df   = load_transaction_data()

    # ------------------------------------------------------------------
    _banner(2, TOTAL, "FEATURE ENGINEERING")
    # ------------------------------------------------------------------
    prime_df     = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)

    # ------------------------------------------------------------------
    _banner(3, TOTAL, "MERGING PRIME + TRANSACTION DATA")
    # ------------------------------------------------------------------
    merged = merge_data(prime_df, txn_features)

    # ------------------------------------------------------------------
    _banner(4, TOTAL, "CREATING TARGET VARIABLE")
    # ------------------------------------------------------------------
    # create_target returns the full DataFrame (with optional sample_weight)
    merged = create_target(merged)
    target = merged[config.TARGET_COL]

    # Extract sample_weight if present (weighted_binary mode), else None
    sample_weight = (
        merged["sample_weight"] if "sample_weight" in merged.columns else None
    )

    if sample:
        print("\n  [SAMPLING MODE] Reducing dataset to 25% via stratified sampling ...")
        keep_idx, _ = train_test_split(
            merged.index,
            train_size=0.25,
            stratify=target,
            random_state=config.RANDOM_STATE,
        )
        merged        = merged.loc[keep_idx]
        target        = target.loc[keep_idx]
        sample_weight = sample_weight.loc[keep_idx] if sample_weight is not None else None

    n_default = int(target.sum())
    n_total   = len(target)
    print(f"  Total samples:   {n_total:,}")
    print(f"  Default     (1): {n_default:,}  ({n_default / n_total * 100:.1f}%)")
    print(f"  Non-default (0): {n_total - n_default:,}  "
          f"({(n_total - n_default) / n_total * 100:.1f}%)")

    customer_ids = merged[config.CUSTOMER_ID].copy()

    # ------------------------------------------------------------------
    _banner(5, TOTAL, "PREPROCESSING (encoding, scaling, imputation)")
    # ------------------------------------------------------------------
    X, y, artifacts = preprocess(merged, target, fit=True)

    # Align sample_weight to X after preprocessing may drop rows
    if sample_weight is not None:
        sample_weight = sample_weight.loc[X.index]

    # ------------------------------------------------------------------
    _banner(6, TOTAL, "LEAKAGE CHECK")
    # ------------------------------------------------------------------
    # Runs before train/test split so we catch leakage on the full feature
    # matrix — any suspect column will have a near-perfect single-feature AUC
    # regardless of which split it's measured on.
    check_for_leakage(X, y, abort_on_leak=True)

    # ------------------------------------------------------------------
    _banner(7, TOTAL, "TRAIN / TEST SPLIT")
    # ------------------------------------------------------------------
    split_kwargs = dict(
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    if sample_weight is not None:
        X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
            X, y, sample_weight, **split_kwargs
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        sw_train = None

    print(f"  Train set:          {X_train.shape[0]:,} samples")
    print(f"  Test set:           {X_test.shape[0]:,} samples")
    print(f"  Features:           {X_train.shape[1]}")
    print(f"  Train default rate: {y_train.sum() / len(y_train) * 100:.1f}%")
    print(f"  Test  default rate: {y_test.sum()  / len(y_test)  * 100:.1f}%")

    # ------------------------------------------------------------------
    if tune:
        _banner(8, TOTAL, "HYPERPARAMETER TUNING + TRAINING")
    else:
        _banner(8, TOTAL, "SMOTE RESAMPLING + XGBOOST TRAINING")
    # ------------------------------------------------------------------

    if tune:
        best_model, best_params = tune_hyperparameters(X_train, y_train)
        print("  Generating predictions on test set ...")
        y_proba = best_model.predict_proba(X_test)[:, 1]

        print("  Finding best decision threshold (F-beta) ...")
        best_threshold = find_best_threshold_fbeta(y_test, y_proba, beta=2.0)
        print(f"  Best threshold: {best_threshold:.4f}")

        y_pred        = (y_proba >= best_threshold).astype(int)
        model_to_save = best_model

    else:
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        # sample_weight rows don't map to SMOTE synthetic rows, so we
        # pass None here ظ¤ SMOTE already balances the class distribution.
        print("  Training XGBoost with early stopping ...")
        bst = train_xgboost(
            X_train_res, y_train_res,
            X_test, y_test,
            sample_weight_train=None,
        )
        print("  Generating predictions on test set ...")
        y_proba = bst.predict_proba(X_test, iteration_range=(0, bst.best_iteration + 1))[:, 1]

        print("  Finding best decision threshold (F-beta) ...")
        best_threshold = find_best_threshold_fbeta(y_test, y_proba, beta=2.0)
        print(f"  Best threshold: {best_threshold:.4f}")

        y_pred        = (y_proba >= best_threshold).astype(int)
        model_to_save = bst

    # ------------------------------------------------------------------
    _banner(9, TOTAL, "EVALUATION & OUTPUT")
    # ------------------------------------------------------------------
    metrics = evaluate(y_test, y_pred, y_proba, threshold=best_threshold)
    report  = generate_report(metrics, y_test, y_pred, config.REPORT_PATH)
    print(report)

    print("  Scoring all customers ...")
    if tune:
        all_proba = best_model.predict_proba(X)[:, 1]
    else:
        all_proba = bst.predict_proba(X, iteration_range=(0, bst.best_iteration + 1))[:, 1]
    all_pred = (all_proba >= best_threshold).astype(int)

    scores_df = pd.DataFrame({
        config.CUSTOMER_ID:    customer_ids.loc[X.index].values,
        "default_probability": all_proba,
        "predicted_label":     all_pred,
    })
    scores_df.to_csv(config.SCORES_PATH, index=False)
    print(f"  Risk scores saved to {config.SCORES_PATH}")

    # Save best_threshold inside artifacts so scoring uses the same cutoff
    save_model(model_to_save, {**artifacts, "best_threshold": best_threshold})

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    try:
        plot_feature_importance()
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Scoring pipeline (new data)
# ---------------------------------------------------------------------------

def run_scoring_pipeline(model_path: str = None,
                         prime_dir:   str = None,
                         txn_dir:     str = None,
                         output_path: str = None):
    """Score new monthly data using a previously trained model."""
    TOTAL = 5
    _ensure_output_dir()
    output_path = output_path or config.SCORES_PATH

    _banner(1, TOTAL, "LOADING SAVED MODEL")
    model, artifacts = load_model(model_path)

    # Use the threshold saved at training time; fall back to 0.5 for
    # models saved before this field was added.
    best_threshold = artifacts.get("best_threshold", 0.5)
    print(f"  Using decision threshold: {best_threshold:.4f}")

    _banner(2, TOTAL, "LOADING NEW DATA")
    prime_df = load_prime_data(prime_dir)
    txn_df   = load_transaction_data(txn_dir)

    _banner(3, TOTAL, "FEATURE ENGINEERING")
    prime_df     = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)
    merged       = merge_data(prime_df, txn_features)

    customer_ids = merged[config.CUSTOMER_ID].copy()

    _banner(4, TOTAL, "PREPROCESSING")
    X, _, _ = preprocess(merged, y=None, fit=False, artifacts=artifacts)

    _banner(5, TOTAL, "SCORING")
    print(f"  Scoring {len(X):,} card accounts ...") 
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))[:, 1]
        except (TypeError, AttributeError):
            proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X, iteration_range=(0, model.best_iteration + 1))

    pred = (proba >= best_threshold).astype(int)     

    scores_df = pd.DataFrame({
        config.CUSTOMER_ID:    customer_ids.values,
        "default_probability": proba,
        "predicted_label":     pred,
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
