#!/usr/bin/env python3
"""
SHAP Analysis Script for Model Interpretability
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def load_data(data_path="data/master_airquality_clean.csv"):
    """Load and prepare dataset."""
    print("📥 Loading data...")
    df = pd.read_csv(data_path, low_memory=False)
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    
    for col in ["PM2.5", "PM10", "O3", "CO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.dropna(subset=["PM2.5"], inplace=True)
    
    FEATURES = ["PM10", "O3", "CO", "hour", "dayofweek", "month"]
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    
    return df, FEATURES


def create_test_split(df, features):
    """Create test split."""
    n = len(df)
    test_size = int(0.2 * n)
    test_df = df.iloc[n - test_size:]
    X_test = test_df[features]
    return X_test


def shap_analysis(model, X_test, output_dir="artifacts"):
    """Perform SHAP analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Computing SHAP values...")
    
    # Use TreeExplainer for tree-based models
    try:
        explainer = shap.TreeExplainer(model)
        # Sample for speed if dataset is large
        X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
    except:
        # Fallback to KernelExplainer for other models
        print("Using KernelExplainer (slower)...")
        background = shap.sample(X_test, 100)
        explainer = shap.KernelExplainer(model.predict, background)
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    print("📊 Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/shap_summary.png")
    
    # Feature importance plot
    print("📊 Generating feature importance plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(f"{output_dir}/shap_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/shap_feature_importance.png")
    
    # Dependence plots for top features
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-3:]  # Top 3 features
    
    for idx in top_features_idx:
        feature_name = X_sample.columns[idx]
        print(f"📊 Generating dependence plot for {feature_name}...")
        plt.figure()
        shap.dependence_plot(idx, shap_values, X_sample, show=False)
        plt.savefig(f"{output_dir}/shap_dependence_{feature_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/shap_dependence_{feature_name}.png")


def main():
    # Load model
    model_path = "models/best_pm25_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📦 Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load data
    df, features = load_data()
    X_test = create_test_split(df, features)
    
    # SHAP analysis
    shap_analysis(model, X_test)
    
    print("\n✅ SHAP analysis complete!")


if __name__ == "__main__":
    main()
