#!/usr/bin/env python3
"""
Training script with Comet ML tracking and model versioning.
"""

import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from comet_ml import Experiment


def load_and_prepare_data(data_path="data/master_airquality_clean.csv"):
    """Load and prepare dataset."""
    print("📥 Loading dataset...")
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


def create_splits(df, features):
    """Create train/test splits."""
    n = len(df)
    test_size = int(0.2 * n)
    
    train_df = df.iloc[:n - test_size]
    test_df = df.iloc[n - test_size:]
    
    X_train = train_df[features]
    y_train = train_df["PM2.5"]
    X_test = test_df[features]
    y_test = test_df["PM2.5"]
    
    return X_train, X_test, y_train, y_test


def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, experiment):
    """Train model and log to Comet ML."""
    experiment.set_name(model_name)
    experiment.add_tag(model_name)
    
    # Log hyperparameters
    if hasattr(model, 'get_params'):
        experiment.log_parameters(model.get_params())
    
    # Train
    print(f"🎯 Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    experiment.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })
    
    print(f"✅ {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    return rmse, model


def main():
    # Initialize Comet ML experiment
    # API key can be set via COMET_API_KEY environment variable
    experiment = Experiment(
        project_name="pm25-airquality",
        workspace=None,  # Will use default workspace
        auto_metric_logging=True,
        auto_param_logging=True,
        log_code=True
    )
    
    # Load data
    df, features = load_and_prepare_data()
    X_train, X_test, y_train, y_test = create_splits(df, features)
    
    print(f"📊 Dataset: Train={len(X_train)}, Test={len(X_test)}")
    
    # Log dataset info
    experiment.log_dataset_info(
        name="master_airquality_clean",
        path="data/master_airquality_clean.csv"
    )
    
    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42
        )
    }
    
    # Train all models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Create new experiment for each model
        exp = Experiment(
            project_name="pm25-airquality",
            workspace=None,
            auto_metric_logging=True
        )
        
        rmse, trained_model = train_and_log_model(
            model, name, X_train, y_train, X_test, y_test, exp
        )
        results[name] = rmse
        trained_models[name] = trained_model
        exp.end()
    
    # Select best model
    best_model_name = min(results, key=results.get)
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name} (RMSE: {results[best_model_name]:.4f})")
    
    # Save best model locally with versioning
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_pm25_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"💾 Saved best model to {model_path}")
    
    # Log model to Comet
    experiment.log_model(best_model_name, model_path)
    experiment.log_other("best_model", best_model_name)
    experiment.log_metric("best_rmse", results[best_model_name])
    experiment.add_tag("production-candidate")
    
    # Save model metadata
    metadata = {
        "model_name": best_model_name,
        "rmse": results[best_model_name],
        "features": features,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open("models/model_metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=4)
    
    experiment.log_asset("models/model_metadata.json")
    experiment.end()
    
    print("\n✨ Training complete! Check Comet ML dashboard for details.")


if __name__ == "__main__":
    main()
