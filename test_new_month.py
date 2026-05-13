"""
Feasibility test: score a new month's data with the saved model and
compare predictions against the actual labels.

Usage
-----
    python test_new_month.py --prime_dir <path> --txn_dir <path>

If no paths are given, it defaults to config.PRIME_DATA_DIR and
config.TRANSACTION_DATA_DIR (useful for a quick sanity check).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_prime_data, load_transaction_data, merge_data
from feature_engineering import (
    engineer_prime_features,
    engineer_transaction_features,
    engineer_temporal_features,
    create_target,
)
from preprocessing import preprocess
from evaluation import evaluate, generate_report


def _banner(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_feasibility_test(prime_dir: str = None,
                         txn_dir: str = None,
                         model_path: str = None):
    """Load new-month data, score with saved model, evaluate vs actual labels."""

    model_path = model_path or config.MODEL_PATH

    # ------------------------------------------------------------------
    _banner("1 / 6  LOADING SAVED MODEL")
    # ------------------------------------------------------------------
    try:
        pipeline_data = joblib.load(model_path)
        model = pipeline_data["model"]
        artifacts = pipeline_data["artifacts"]
    except Exception as e:
        print(f"  ERROR: Could not load model from {model_path}: {e}")
        return None

    best_threshold = artifacts.get("best_threshold", 0.5)
    selected_features = artifacts.get("selected_features", None)
    print(f"  Model loaded from {model_path}")
    print(f"  Decision threshold: {best_threshold:.4f}")
    if selected_features:
        print(f"  Selected features:  {len(selected_features)}")

    # ------------------------------------------------------------------
    _banner("2 / 6  LOADING NEW MONTH DATA")
    # ------------------------------------------------------------------
    prime_df = load_prime_data(prime_dir)
    txn_df = load_transaction_data(txn_dir)

    # ------------------------------------------------------------------
    _banner("3 / 6  FEATURE ENGINEERING")
    # ------------------------------------------------------------------
    prime_df = engineer_prime_features(prime_df)
    txn_features = engineer_transaction_features(txn_df)
    merged = merge_data(prime_df, txn_features)

    # ------------------------------------------------------------------
    _banner("4 / 6  TARGET CREATION")
    # ------------------------------------------------------------------
    # Since labels exist in the new data, create the target for evaluation
    merged = create_target(merged)
    merged = engineer_temporal_features(merged)
    target = merged[config.TARGET_COL]

    n_default = int(target.sum())
    n_total = len(target)
    print(f"  Total samples:   {n_total:,}")
    print(f"  Default     (1): {n_default:,}  ({n_default / n_total * 100:.1f}%)")
    print(f"  Non-default (0): {n_total - n_default:,}  "
          f"({(n_total - n_default) / n_total * 100:.1f}%)")

    # ------------------------------------------------------------------
    _banner("5 / 6  PREPROCESSING & SCORING")
    # ------------------------------------------------------------------
    X, y, _ = preprocess(merged, target, fit=False, artifacts=artifacts)

    # Subset to selected features if the model was trained with feature selection
    if selected_features:
        missing = [f for f in selected_features if f not in X.columns]
        if missing:
            print(f"  WARNING: {len(missing)} selected features missing from new data: {missing}")
        available = [f for f in selected_features if f in X.columns]
        X = X[available]
        print(f"  Using {len(available)} / {len(selected_features)} selected features")

    # Generate predictions
    import xgboost as xgb
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        y_proba = model.predict(dmatrix)

    y_pred = (y_proba >= best_threshold).astype(int)

    # ------------------------------------------------------------------
    _banner("6 / 6  EVALUATION")
    # ------------------------------------------------------------------
    metrics = evaluate(y, y_pred, y_proba, threshold=best_threshold)

    # Build model params dict for the report
    model_params = {
        "model_path": os.path.basename(model_path),
        "best_threshold": round(best_threshold, 4),
        "n_features": X.shape[1],
    }

    report = generate_report(
        metrics, y, y_pred,
        output_path=os.path.join(config.OUTPUT_DIR, "feasibility_report.txt"),
        model_params=model_params,
    )
    print(report)

    # --- Per-status breakdown ---
    if config.STATUS_COL in merged.columns:
        print()
        print("PER-STATUS BREAKDOWN")
        print("-" * 60)
        status_df = pd.DataFrame({
            "status": merged.loc[X.index, config.STATUS_COL].values,
            "actual": y.values,
            "predicted": y_pred,
            "probability": y_proba,
        })
        for status, grp in status_df.groupby("status"):
            n = len(grp)
            n_flagged = int(grp["predicted"].sum())
            avg_prob = grp["probability"].mean()
            actual_default = int(grp["actual"].sum())
            print(f"  {status:<8s}  n={n:>6,}  "
                  f"actual_default={actual_default:>5,}  "
                  f"predicted_default={n_flagged:>5,}  "
                  f"avg_prob={avg_prob:.3f}")

    print()
    print("=" * 60)
    print("  FEASIBILITY TEST COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feasibility test: score new month data and evaluate vs actual labels."
    )
    parser.add_argument(
        "--prime_dir", type=str, default=None,
        help="Directory containing new-month prime CSV(s). "
             "Defaults to config.PRIME_DATA_DIR.",
    )
    parser.add_argument(
        "--txn_dir", type=str, default=None,
        help="Directory containing new-month transaction CSV(s). "
             "Defaults to config.TRANSACTION_DATA_DIR.",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to saved model .joblib. Defaults to config.MODEL_PATH.",
    )
    args = parser.parse_args()

    run_feasibility_test(
        prime_dir=args.prime_dir,
        txn_dir=args.txn_dir,
        model_path=args.model_path,
    )
