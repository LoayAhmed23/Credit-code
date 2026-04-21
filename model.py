"""
Model training, SMOTE resampling, hyperparameter tuning, and persistence.
"""

import numpy as np
import lightgbm as lgb
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

import config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_val, y_val,
                   params=None, sample_weight_train=None):
    """Train a LightGBM model with early stopping.

    Returns the trained Booster.
    """
    params = params or config.LGBM_PARAMS

    # Inject scale_pos_weight if we know it's a very imbalanced dataset
    # We estimate missing parameters based on ~6.4% rate, so negative ~ 93.6%
    if params and "scale_pos_weight" not in params and "is_unbalance" not in params:
        # Default ~ 6.4%, non-default ~ 93.6%, ratio is about 14.6
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count > 0:
            params["scale_pos_weight"] = neg_count / pos_count

    lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weight_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)

    bst = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=10),
        ],
    )
    best_auc = bst.best_score.get("valid", {}).get("auc", "N/A")
    print(f"  Best iteration: {bst.best_iteration}  |  Best valid AUC: {best_auc}")
    return bst


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Run RandomizedSearchCV over LightGBM pipeline.

    Returns (best_estimator, best_params).
    """
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    # SMOTE is being removed, so ImbPipeline not strictly necessary, 
    # but we will just pass LGBMClassifier directly if we drop SMOTE.
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        verbosity=-1,
        random_state=config.RANDOM_STATE,
        scale_pos_weight=spw,
    )

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Convert parameter grid to avoid model__ prefix since we don't have pipeline
    param_grid = {k.replace("model__", ""): v for k, v in config.TUNE_PARAM_GRID.items()}

    search = RandomizedSearchCV(
        model,
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
