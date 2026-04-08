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
# SMOTE
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to the training set."""
    sm = SMOTE(sampling_strategy="auto", random_state=config.RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(
        f"[model] SMOTE: {X_train.shape[0]} -> {X_res.shape[0]} samples  "
        f"(minority: {int(y_train.sum())} -> {int(y_res.sum())})"
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train a LightGBM model with early stopping.

    Returns the trained Booster.
    """
    params = params or config.LGBM_PARAMS

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    bst = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"[model] Best iteration: {bst.best_iteration}")
    return bst


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Run RandomizedSearchCV over a SMOTE + LightGBM pipeline.

    Returns (best_estimator, best_params).
    """
    pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="binary",
                    metric="auc",
                    boosting_type="gbdt",
                    verbosity=-1,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=config.TUNE_PARAM_GRID,
        n_iter=config.TUNE_N_ITER,
        scoring=scorer,
        cv=config.TUNE_CV_FOLDS,
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print("[model] Starting hyperparameter search ...")
    search.fit(X_train, y_train)

    print(f"[model] Best CV AUC: {search.best_score_:.4f}")
    print(f"[model] Best params: {search.best_params_}")
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
