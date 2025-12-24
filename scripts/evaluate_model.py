#!/usr/bin/env python3
"""
Model Evaluation Script
Generates comprehensive evaluation report with metrics, plots, and error analysis.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error


def load_data(data_path="data/master_airquality_clean.csv"):
    """Load and prepare the dataset."""
    print(f"📥 Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Prepare timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    
    # Extract time features
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    
    # Convert pollutants to numeric
    for col in ["PM2.5", "PM10", "O3", "CO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.dropna(subset=["PM2.5"], inplace=True)
    
    FEATURES = ["PM10", "O3", "CO", "hour", "dayofweek", "month"]
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    
    return df, FEATURES


def create_splits(df, features, target="PM2.5"):
    """Create train/test splits."""
    n = len(df)
    test_size = int(0.2 * n)
    
    train_df = df.iloc[:n - test_size]
    test_df = df.iloc[n - test_size:]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    return X_train, X_test, y_train, y_test, test_df


def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    return metrics


def plot_predictions(y_true, y_pred, output_dir="artifacts"):
    """Generate prediction vs actual plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=1)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Predicted vs Actual PM2.5")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pred_vs_actual.png", dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/pred_vs_actual.png")


def plot_residuals(y_true, y_pred, output_dir="artifacts"):
    """Generate residual plots."""
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual distribution
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Residuals")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Residuals")
    axes[0].axvline(0, color='r', linestyle='--')
    
    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=1)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Predicted PM2.5")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals.png", dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/residuals.png")


def error_analysis(y_true, y_pred, test_df, output_dir="artifacts"):
    """Perform error analysis on worst predictions."""
    residuals = np.abs(y_pred - y_true)
    test_df = test_df.copy()
    test_df["predictions"] = y_pred
    test_df["residuals"] = residuals
    
    # Top 10 worst predictions
    worst = test_df.nlargest(10, "residuals")[["Timestamp", "PM2.5", "predictions", "residuals"]]
    worst.to_csv(f"{output_dir}/worst_predictions.csv", index=False)
    print(f"✅ Saved: {output_dir}/worst_predictions.csv")
    
    # Error by hour of day
    hourly_error = test_df.groupby("hour")["residuals"].mean()
    
    plt.figure(figsize=(10, 5))
    hourly_error.plot(kind='bar')
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Absolute Error")
    plt.title("Error Analysis by Hour of Day")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_by_hour.png", dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/error_by_hour.png")


def save_metrics(metrics, output_path="artifacts/evaluation_metrics.json"):
    """Save metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Saved: {output_path}")


def main():
    # Load model
    model_path = "models/best_pm25_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📦 Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load and split data
    df, features = load_data()
    X_train, X_test, y_train, y_test, test_df = create_splits(df, features)
    
    print(f"📊 Dataset split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Make predictions
    print("🔮 Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("📈 Evaluating model...")
    metrics = evaluate_metrics(y_test, y_pred)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:8s}: {value:.4f}")
    print("="*50 + "\n")
    
    # Generate plots
    print("📊 Generating visualizations...")
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    error_analysis(y_test, y_pred, test_df)
    
    # Save metrics
    save_metrics(metrics)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
