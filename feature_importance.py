import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
import joblib

sys.path.append(os.path.abspath('.'))

from pipeline import run_training_pipeline
import config

def plot_feature_importance():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 10)

    print("Starting XGBoost pipeline on 25% stratified partition...")
    metrics = run_training_pipeline(tune=False, sample=True)
    print("\n--- Pipeline Finished. Model saved explicitly ---")

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
        try:
            feature_names = model.get_booster().feature_names
        except:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    print(f"Total features fed to XGBoost: {len(feature_names)}")

    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        importance_gain = booster.get_score(importance_type='gain')
        importance_weight = booster.get_score(importance_type='weight') # Equivalent to split

        # Maps dict mapping string to numerical logic back to correctly ordered list matching feature_order
        gain_list = [importance_gain.get(f, 0) for f in feature_names]
        weight_list = [importance_weight.get(f, 0) for f in feature_names]
        
        df_imp = pd.DataFrame({
            'Feature Name': feature_names,
            'Importance (Gain)': gain_list,
            'Importance (Split)': weight_list
        })

        df_imp['Feature Name'] = df_imp['Feature Name'].str.replace('num__', '').str.replace('cat__', '')
        
        df_imp_gain = df_imp.sort_values(by='Importance (Gain)', ascending=False).head(30)
        df_imp_split = df_imp.sort_values(by='Importance (Split)', ascending=False).head(30)

        output_dir = os.path.join(config.BASE_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 12))
        sns.barplot(x='Importance (Gain)', y='Feature Name', data=df_imp_gain, palette='viridis')
        plt.title('Top 30 Feature Importances (Gain - How much the feature reduced tree loss)', fontsize=14)
        plt.xlabel('Gain Importance', fontsize=12)
        plt.ylabel('Engineered Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance_gain.png"))
        plt.close()
        
        plt.figure(figsize=(12, 12))
        sns.barplot(x='Importance (Split)', y='Feature Name', data=df_imp_split, palette='magma')
        plt.title('Top 30 Feature Importances (Split - How many times feature was used for decision)', fontsize=14)
        plt.xlabel('Split Importance', fontsize=12)
        plt.ylabel('Engineered Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance_split.png"))
        plt.close()
        print(f"Saved feature importance plots to {output_dir}")
    else:
        print("Model doesn't support feature importance.")

if __name__ == "__main__":
    plot_feature_importance()
