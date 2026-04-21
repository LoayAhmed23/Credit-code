"""
Model training, hyperparameter tuning, and persistence.
"""

import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

import config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val, params=None, sample_weight_train=None):
    """Train an XGBoost model with early stopping.

    Returns the trained Booster.
    """
    params = params or config.XGB_PARAMS

    # Automatic scale_pos_weight calculation if not provided
    if "scale_pos_weight" not in params:
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        params["scale_pos_weight"] = num_neg / num_pos if num_pos > 0 else 1.0

    xgb_train = xgb.DMatrix(X_train, label=y_train, weight=sample_weight_train)
    xgb_val   = xgb.DMatrix(X_val,   label=y_val)

    bst = xgb.train(
        params,
        xgb_train,
        num_boost_round=config.NUM_BOOST_ROUND,      
        evals=[(xgb_train, "train"), (xgb_val, "valid")],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose_eval=10,
    )
    
    best_auc = bst.best_score
    print(f"  Best iteration: {bst.best_iteration}  |  Best valid AUC: {best_auc}")
    return bst


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Run RandomizedSearchCV over XGBoost.

    Returns (best_estimator, best_params).
    """

    # Automatic scale_pos_weight calculation
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=config.RANDOM_STATE,
        scale_pos_weight=scale_pos_weight
    )

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=config.TUNE_PARAM_GRID,  
        n_iter=config.TUNE_N_ITER,
        scoring=scorer,
        cv=config.TUNE_CV_FOLDS,
        verbose=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1, # Parallelize cross-validation folds
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
