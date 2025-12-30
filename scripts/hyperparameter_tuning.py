# Hyperparameter Tuning Script
# Optimize model hyperparameters using GridSearchCV

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(data_path="data/master_airquality_clean.csv", sample_size=500_000, random_state=42):
    """Load and prepare data with optional downsampling to avoid OOM."""
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path, low_memory=False)

    # Downsample if extremely large to keep CV feasible
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"⚡ Downsampled to {len(df)} rows for faster tuning")
    
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
    """Create train/val/test splits."""
    n = len(df)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    X_train = train_df[features]
    y_train = train_df["PM2.5"]
    X_val = val_df[features]
    y_val = val_df["PM2.5"]
    X_test = test_df[features]
    y_test = test_df["PM2.5"]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_random_forest(X_train, y_train, X_val, y_val):
    """Tune Random Forest hyperparameters with a lean search space to reduce memory/time."""
    print("\n" + "="*60)
    print("TUNING RANDOM FOREST")
    print("="*60)
    
    param_grid = {
        'n_estimators': [75, 125, 175],
        'max_depth': [10, 16, 24],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'max_samples': [0.5, 0.7]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Use RandomizedSearchCV for faster search
    random_search = RandomizedSearchCV(
        rf, param_grid, n_iter=10, cv=2, 
        scoring='neg_mean_squared_error',
        random_state=42, verbose=2, n_jobs=-1
    )
    
    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"\n✅ Best Parameters: {random_search.best_params_}")
    print(f"✅ Validation RMSE: {rmse:.4f}")
    
    return best_model, random_search.best_params_


def tune_xgboost(X_train, y_train, X_val, y_val):
    """Tune XGBoost hyperparameters with constrained search for speed."""
    print("\n" + "="*60)
    print("TUNING XGBOOST")
    print("="*60)
    
    param_grid = {
        'n_estimators': [120, 200, 260],
        'learning_rate': [0.03, 0.05, 0.08],
        'max_depth': [4, 6, 8],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3]
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42
    )
    
    # Use RandomizedSearchCV
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=12, cv=2,
        scoring='neg_mean_squared_error',
        random_state=42, verbose=2, n_jobs=-1
    )
    
    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"\n✅ Best Parameters: {random_search.best_params_}")
    print(f"✅ Validation RMSE: {rmse:.4f}")
    
    return best_model, random_search.best_params_


def evaluate_final_model(model, X_test, y_test):
    """Evaluate best model on test set."""
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for PM2.5 models")
    parser.add_argument("--sample-size", type=int, default=500_000, help="Rows to sample for tuning (set None for full dataset)")
    args = parser.parse_args()

    print("="*70)
    print(" "*20 + "HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data (downsample to avoid OOM during CV)
    df, features = load_data(sample_size=args.sample_size)
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(df, features)
    
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Tune models
    rf_model, rf_params = tune_random_forest(X_train, y_train, X_val, y_val)
    xgb_model, xgb_params = tune_xgboost(X_train, y_train, X_val, y_val)
    
    # Compare on test set
    print("\n" + "="*60)
    print("COMPARING MODELS ON TEST SET")
    print("="*60)
    
    rf_metrics = evaluate_final_model(rf_model, X_test, y_test)
    xgb_metrics = evaluate_final_model(xgb_model, X_test, y_test)
    
    # Select best model
    best_model_name = "RandomForest" if rf_metrics["rmse"] < xgb_metrics["rmse"] else "XGBoost"
    best_model = rf_model if best_model_name == "RandomForest" else xgb_model
    best_params = rf_params if best_model_name == "RandomForest" else xgb_params
    
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Save
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_pm25_model_tuned.pkl")
    
    with open("models/tuning_results.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "best_params": best_params,
            "random_forest_metrics": rf_metrics,
            "xgboost_metrics": xgb_metrics
        }, f, indent=4)
    
    print("\n✅ Tuning complete!")
    print(f"   Model saved: models/best_pm25_model_tuned.pkl")
    print(f"   Results saved: models/tuning_results.json")


if __name__ == "__main__":
    main()
