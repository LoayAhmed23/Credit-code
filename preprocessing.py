"""
Preprocessing: missing-value imputation, encoding, and scaling.
Supports fit (training) and transform-only (scoring) modes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def preprocess(X: pd.DataFrame, y: pd.Series = None,
               fit: bool = True, artifacts: dict = None):
    """Preprocess features for modelling.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix (after feature engineering, before modelling).
    y : pd.Series, optional
        Target vector.  Only needed when ``fit=True`` to align indices.
    fit : bool
        If True, fit encoders / scaler and return them in *artifacts*.
        If False, reuse the encoders / scaler from *artifacts*.
    artifacts : dict, optional
        Previously fitted {encoders, scaler, cat_cols, num_cols, feature_order}.

    Returns
    -------
    X_out : pd.DataFrame
    y_out : pd.Series  (or None when y is None)
    artifacts : dict
    """
    X = X.copy()

    # --- Drop columns that should not be features ---
    cols_to_drop = [c for c in config.DROP_COLS if c in X.columns]
    X = X.drop(columns=cols_to_drop, errors="ignore")

    if fit:
        # Discover column types
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        # --- Missing-value imputation ---
        medians = {}
        modes = {}
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            med = X[c].median()
            medians[c] = med
            X[c] = X[c].fillna(med)
        for c in cat_cols:
            mode_val = X[c].mode().iloc[0] if not X[c].mode().empty else "UNKNOWN"
            modes[c] = mode_val
            X[c] = X[c].fillna(mode_val)

        # --- Label-encode categoricals ---
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            encoders[c] = le

        # --- Standard-scale numericals ---
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        artifacts = {
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "encoders": encoders,
            "scaler": scaler,
            "medians": medians,
            "modes": modes,
            "feature_order": X.columns.tolist(),
        }
    else:
        # --- Transform mode: reuse fitted artifacts ---
        if artifacts is None:
            raise ValueError("artifacts must be provided when fit=False")

        cat_cols = artifacts["cat_cols"]
        num_cols = artifacts["num_cols"]

        for c in num_cols:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X[c] = X[c].fillna(artifacts["medians"].get(c, 0))
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].fillna(artifacts["modes"].get(c, "UNKNOWN"))

        for c in cat_cols:
            if c in X.columns:
                le = artifacts["encoders"][c]
                # Handle unseen labels gracefully
                X[c] = X[c].astype(str).map(
                    lambda v, _le=le: (
                        _le.transform([v])[0] if v in _le.classes_ else -1
                    )
                )

        existing_num = [c for c in num_cols if c in X.columns]
        if existing_num:
            X[existing_num] = artifacts["scaler"].transform(X[existing_num])

        # Ensure same column order; add missing columns as 0
        for c in artifacts["feature_order"]:
            if c not in X.columns:
                X[c] = 0
        X = X[artifacts["feature_order"]]

    # Align indices
    if y is not None:
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]

    print(f"[preprocess] Output shape: {X.shape}  |  fit={fit}")
    return X, y, artifacts
