"""
Model training, SMOTE resampling, hyperparameter tuning, and persistence.
"""

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

import config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val,   
                   params=None, sample_weight_train=None):
    """Train an XGBoost model with early stopping.   

    Returns the trained Booster.
    """
    params = params or config.XGB_PARAMS

    # XGBoost scikit-learn API seamlessly integrates with native DMatrix backend internally
    # and provides better GridSearchCV synergy than the native training API block
    clf = xgb.XGBClassifier(
        n_estimators=config.NUM_BOOST_ROUND,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        **params
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight=sample_weight_train,
        verbose=10
    )
    
    best_auc = clf.best_score
    print(f"  Best iteration: {clf.best_iteration}  |  Best valid AUC: {best_auc}")
    return clf


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Run RandomizedSearchCV over a SMOTE + XGBoost pipeline.

    Returns (best_estimator, best_params).
    """
    pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            (
                "model",                          # step name is "model"
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    device="cuda",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Convert parameter grid to avoid model__ prefix since we don't have pipeline
    param_grid = {k.replace("model__", ""): v for k, v in config.TUNE_PARAM_GRID.items()}

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=config.TUNE_N_ITER,
        scoring=scorer,
        cv=config.TUNE_CV_FOLDS,
        verbose=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        f"  Starting hyperparameter search "
        f"({config.TUNE_N_ITER} iterations x {config.TUNE_CV_FOLDS}-fold CV) ..."
    )
    search.fit(X_train, y_train)

    print(f"  Best CV AUC: {search.best_score_:.4f}")
    print("  Best params:")
    for k, v in search.best_params_.items():
        print(f"    {k}: {v}")

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
