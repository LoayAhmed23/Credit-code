import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

sys.path.append(os.path.abspath('.'))

import config

def plot_feature_importance():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 10)

    # Note: We no longer run the pipeline from here to avoid circular imports.
    # The pipeline will simply call this function after it finishes and saves the model.

    try:
        pipeline_data = joblib.load(config.MODEL_PATH)  
        model = pipeline_data["model"]
        artifacts = pipeline_data["artifacts"]
        print("Loaded model and artifacts successfully.")
    except Exception as e:
        print(f"Could not load the model from {config.MODEL_PATH}: {e}")
        return

    try:
        feature_names = artifacts["feature_order"]
    except Exception as e:
        print(f"Could not extract feature names: {e}")
        if hasattr(model, "feature_names"):
            feature_names = model.feature_names
        elif hasattr(model, "get_score"):
            feature_names = list(model.get_score().keys())
        else:
            print("Fallback for features failed.")
            return

    print(f"Total features fed to XGBoost: {len(feature_names)}")

    if hasattr(model, "get_score"):
        try:
            # For XGBoost Booster, get_score accepts importance_type='gain' / 'weight'
            importance_gain_dict = model.get_score(importance_type="gain")
            importance_weight_dict = model.get_score(importance_type="weight")

            df_imp = pd.DataFrame({
                'Feature Name': feature_names
            })

            df_imp['Importance (Gain)'] = df_imp['Feature Name'].map(importance_gain_dict).fillna(0)
            df_imp['Importance (Weight)'] = df_imp['Feature Name'].map(importance_weight_dict).fillna(0)
            df_imp['Feature Name'] = df_imp['Feature Name'].str.replace('num__', '').str.replace('cat__', '')

            df_imp_gain = df_imp.sort_values(by='Importance (Gain)', ascending=False).head(30)
            df_imp_weight = df_imp.sort_values(by='Importance (Weight)', ascending=False).head(30)

            output_dir = os.path.join(config.BASE_DIR, "output")
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(12, 12))
            sns.barplot(x='Importance (Gain)', y='Feature Name', data=df_imp_gain, palette='viridis')
            plt.title('Top 30 Feature Importances (Gain)', fontsize=14)
            plt.xlabel('Gain Importance', fontsize=12)
            plt.ylabel('Engineered Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance_gain.png"))
            plt.close()

            plt.figure(figsize=(12, 12))
            sns.barplot(x='Importance (Weight)', y='Feature Name', data=df_imp_weight, palette='magma')
            plt.title('Top 30 Feature Importances (Weight / Splits)', fontsize=14)
            plt.xlabel('Weight Importance', fontsize=12)
            plt.ylabel('Engineered Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance_split.png"))
            plt.close()

            print(f"Saved feature importance plots to {output_dir}")
        except Exception as e:
            print(f"Error extracting importance scores: {e}")
    else:
        print("Model doesn't support feature importance via get_score.")


def plot_feature_target_correlation(X: pd.DataFrame = None,
                                    y: pd.Series = None,
                                    top_n: int = 30):
    """Plot the Pearson correlation between each feature and the binary target.

    Parameters
    ----------
    X : pd.DataFrame, optional
        Preprocessed feature matrix.  If *None*, the function will
        re-derive it from the raw data using the saved model artifacts.
    y : pd.Series, optional
        Binary target vector.  Required when *X* is provided.
    top_n : int
        Number of features to display (by absolute correlation).
    """
    output_dir = os.path.join(config.BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # If X/y not supplied, reconstruct from raw data + saved artifacts
    # ------------------------------------------------------------------
    if X is None or y is None:
        from data_loader import load_prime_data, load_transaction_data, merge_data
        from feature_engineering import (
            engineer_prime_features,
            engineer_transaction_features,
            engineer_temporal_features,
            create_target,
        )
        from preprocessing import preprocess

        print("  [correlation] Re-building X/y from raw data ...")
        prime_df = load_prime_data()
        txn_df = load_transaction_data()
        prime_df = engineer_prime_features(prime_df)
        txn_features = engineer_transaction_features(txn_df)
        merged = merge_data(prime_df, txn_features)
        merged = create_target(merged)
        merged = engineer_temporal_features(merged)
        target = merged[config.TARGET_COL]

        try:
            pipeline_data = joblib.load(config.MODEL_PATH)
            artifacts = pipeline_data["artifacts"]
        except Exception as e:
            print(f"  Could not load artifacts for correlation plot: {e}")
            return

        X, y, _ = preprocess(merged, target, fit=False, artifacts=artifacts)

    # ------------------------------------------------------------------
    # Compute correlations
    # ------------------------------------------------------------------
    # Keep only numeric columns (label-encoded cats are already int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr_values = X[numeric_cols].corrwith(y).dropna()
    corr_values.name = "correlation"

    # Sort by absolute correlation and keep top N
    corr_df = (
        corr_values
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .to_frame(name="abs_corr")
    )
    # Attach the signed value for the colour mapping
    corr_df["correlation"] = corr_values.loc[corr_df.index]
    corr_df["Feature"] = (
        corr_df.index
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))

    palette = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_df["correlation"]]

    sns.barplot(
        x="correlation", y="Feature", data=corr_df,
        palette=palette, orient="h", ax=ax,
    )

    ax.set_title(
        f"Top {len(corr_df)} Feature–Target Correlations (Pearson)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Pearson Correlation with Target", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_target_correlation.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved feature–target correlation plot to {save_path}")


if __name__ == "__main__":
    plot_feature_importance()
    plot_feature_target_correlation()
