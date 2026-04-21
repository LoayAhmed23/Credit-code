import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def train_intersection_model():
    """Extract top 20 features by Gain and Top 30 by Weight, find their intersection, 
    and train a faster, focused model."""
    print("\n" + "="*60)
    print("  TRAINING NEW MODEL ON FEATURE INTERSECTION")
    print("="*60)
    
    try:
        pipeline_data = joblib.load(config.MODEL_PATH)  
        model = pipeline_data["model"]
        artifacts = pipeline_data["artifacts"]
    except Exception as e:
        print(f"Base model required to extract importances not found: {e}")
        return

    try:
        feature_names = artifacts["feature_order"]      
    except Exception as e:
        if hasattr(model, "feature_names"):
            feature_names = model.feature_names
        elif hasattr(model, "get_score"):
            feature_names = list(model.get_score().keys())
        else:
            return

    if not hasattr(model, "get_score"):
        print("Base model does not support get_score()")
        return

    # Extract raw importance dicts
    importance_gain_dict = model.get_score(importance_type="gain")
    importance_weight_dict = model.get_score(importance_type="weight")

    # Map to DataFrame
    df_imp = pd.DataFrame({'Feature Name': feature_names})
    df_imp['Importance (Gain)'] = df_imp['Feature Name'].map(importance_gain_dict).fillna(0)
    df_imp['Importance (Weight)'] = df_imp['Feature Name'].map(importance_weight_dict).fillna(0)        

    # Clean names (required since pipeline uses cleaned names or native columns)
    df_imp['Clean Name'] = df_imp['Feature Name'].str.replace('num__', '').str.replace('cat__', '')   

    # Extract top 20 Gain and Top 30 weight specifically by name
    top_20_gain = set(df_imp.sort_values(by='Importance (Gain)', ascending=False).head(20)['Clean Name'])
    top_30_weight = set(df_imp.sort_values(by='Importance (Weight)', ascending=False).head(30)['Clean Name'])

    intersection_features = list(top_20_gain.intersection(top_30_weight))

    print("\n" + "*"*60)
    print("  COPY-PASTE READY INTERSECTION LIST:")
    print("*"*60)
    print(f"INTERSECTION_FEATURES = {intersection_features}")
    print("*"*60 + "\n")

    if not intersection_features:
        print("No intersecting features found. Cannot train model.")
        return

    # To avoid overwriting the base model, we can temporarily change the save path in config
    original_model_path = config.MODEL_PATH
    config.MODEL_PATH = original_model_path.replace(".joblib", "_intersection.joblib")
    
    try:
        from pipeline import run_training_pipeline
        # Run training loop *specifically* on the finalized intersection
        metrics = run_training_pipeline(tune=False, sample=False, selected_features=intersection_features)
        print(f"\nFinal Intersection Model Metrics: {metrics}")
    except Exception as e:
        print(f"Failed to train intersection model: {e}")
    finally:
        # Restore configuration
        config.MODEL_PATH = original_model_path


if __name__ == "__main__":
    plot_feature_importance()
    train_intersection_model()
