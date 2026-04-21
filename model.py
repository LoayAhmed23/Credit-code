"""
Model training, hyperparameter tuning, and persistence.
"""

import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

import config


def get_top_features_by_gain(booster: "xgb.Booster", feature_order: list[str], top_n: int = 20) -> list[str]:
    """Return the top-N features ranked by XGBoost 'gain'.

    Notes
    -----
    - Only features present in `feature_order` are considered.
    - Features never used in a split have gain=0 and will be ranked last.
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    gain_dict = booster.get_score(importance_type="gain")

    scored = [(f, float(gain_dict.get(f, 0.0))) for f in feature_order]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [f for f, _ in scored[: min(top_n, len(scored))]]


def subset_to_features(X_train, X_test, X_all, artifacts: dict, selected_features: list[str]):
    """Subset train/test/all matrices + artifacts to a fixed ordered feature list."""
    if not selected_features:
        raise ValueError("selected_features is empty")

    # Keep only features that exist (defensive against any mismatch)
    selected_features = [c for c in selected_features if c in X_train.columns]
    if not selected_features:
        raise ValueError("None of the selected_features exist in X_train")

    X_train2 = X_train[selected_features]
    X_test2 = X_test[selected_features]
    X_all2 = X_all[selected_features]

    artifacts2 = {**artifacts, "feature_order": list(selected_features)}
    return X_train2, X_test2, X_all2, artifacts2


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

    # Use the number of available cores for faster Parallel CV tuning.
    import multiprocessing
    n_cores = multiprocessing.cpu_count()

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=config.TUNE_PARAM_GRID,  
        n_iter=config.TUNE_N_ITER,
        scoring=scorer,
        cv=config.TUNE_CV_FOLDS,
        verbose=2,
        random_state=config.RANDOM_STATE,
        n_jobs=config.n_gpus, # Parallelize cross-validation folds across GPUs, not CPUs to avoid OOM
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
