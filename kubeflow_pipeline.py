"""
Kubeflow Pipeline for PM2.5 Air Quality Prediction
Orchestrates the complete ML workflow using Kubeflow Pipelines
"""

import kfp
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    InputPath,
    OutputPath
)


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "numpy<2.0",
        "pyarrow"
    ],
    base_image="python:3.10"
)
def data_ingestion_component(
    output_data: Output[Dataset],
    data_path: str = "data/kaggle_csvs"
):
    """Ingest and combine raw CSV files"""
    import pandas as pd
    import os
    import glob
    from datetime import datetime
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"📥 DATA INGESTION COMPONENT - Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"🔍 Data path: {data_path}")
    print(f"🔍 Absolute path: {os.path.abspath(data_path) if os.path.exists(data_path) else 'PATH NOT FOUND'}")
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"\n📊 Found {len(csv_files)} CSV files to process")
    
    if len(csv_files) == 0:
        print(f"⚠️  WARNING: No CSV files found in {data_path}")
        print(f"   Please verify the data path is correct")
    
    dfs = []
    failed_files = []
    
    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
            if (i + 1) % 50 == 0:
                print(f"   ✓ Processed {i + 1}/{len(csv_files)} files ({((i+1)/len(csv_files)*100):.1f}%)")
        except Exception as e:
            print(f"   ✗ Error reading {os.path.basename(file)}: {str(e)[:100]}")
            failed_files.append(os.path.basename(file))
    
    print(f"\n📈 Processing Summary:")
    print(f"   ✓ Successfully loaded: {len(dfs)} files")
    print(f"   ✗ Failed: {len(failed_files)} files")
    if failed_files:
        print(f"   Failed files: {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Combined dataset: {len(combined_df):,} rows × {len(combined_df.columns)} columns")
    print(f"   Columns: {', '.join(combined_df.columns[:10])}{'...' if len(combined_df.columns) > 10 else ''}")
    print(f"   Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save to output
    combined_df.to_csv(output_data.path, index=False)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n💾 Data saved to: {output_data.path}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"{'='*80}\n")


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "numpy<2.0",
        "scikit-learn"
    ],
    base_image="python:3.10"
)
def data_preprocessing_component(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    test_size: float = 0.2
):
    """Clean and preprocess the data"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"🧹 DATA PREPROCESSING COMPONENT - Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv(input_data.path, low_memory=False)
    initial_rows = len(df)
    print(f"\n📊 Initial dataset: {initial_rows:,} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Parse timestamps
    print(f"\n⏰ Parsing timestamps...")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    invalid_timestamps = df["Timestamp"].isna().sum()
    print(f"   Invalid timestamps: {invalid_timestamps:,} ({invalid_timestamps/initial_rows*100:.2f}%)")
    df.dropna(subset=["Timestamp"], inplace=True)
    print(f"   Rows after timestamp cleaning: {len(df):,}")
    
    # Extract temporal features
    print(f"\n🔧 Extracting temporal features...")
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    print(f"   ✓ Added features: hour, dayofweek, month")
    
    # Convert numeric columns
    print(f"\n🔢 Converting numeric columns...")
    numeric_cols = ["PM2.5", "PM10", "O3", "CO"]
    for col in numeric_cols:
        before = df[col].isna().sum() if col in df.columns else len(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        print(f"   {col}: {after:,} missing values ({after/len(df)*100:.2f}%)")
    
    # Remove rows with missing target
    print(f"\n🎯 Handling target variable (PM2.5)...")
    before_target = len(df)
    df.dropna(subset=["PM2.5"], inplace=True)
    removed = before_target - len(df)
    print(f"   Removed {removed:,} rows with missing PM2.5 ({removed/before_target*100:.2f}%)")
    
    # Fill missing features with median
    print(f"\n🔧 Filling missing feature values with median...")
    features = ["PM10", "O3", "CO", "hour", "dayofweek", "month"]
    for feat in features:
        missing = df[feat].isna().sum()
        if missing > 0:
            median_val = df[feat].median()
            print(f"   {feat}: filling {missing:,} values with median {median_val:.2f}")
    df[features] = df[features].fillna(df[features].median())
    
    # Keep only required columns
    df = df[["PM2.5"] + features]
    
    # Data quality summary
    print(f"\n📈 Preprocessing Summary:")
    print(f"   Initial rows: {initial_rows:,}")
    print(f"   Final rows: {len(df):,}")
    print(f"   Data retained: {len(df)/initial_rows*100:.2f}%")
    print(f"   PM2.5 range: [{df['PM2.5'].min():.2f}, {df['PM2.5'].max():.2f}]")
    print(f"   PM2.5 mean: {df['PM2.5'].mean():.2f} µg/m³")
    print(f"   Missing values: {df.isna().sum().sum()}")
    
    # Save processed data
    df.to_csv(output_data.path, index=False)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n💾 Preprocessed data saved to: {output_data.path}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"✅ Preprocessing complete!")
    print(f"{'='*80}\n")


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "numpy<2.0",
        "scikit-learn",
        "xgboost",
        "joblib"
    ],
    base_image="python:3.10"
)
def train_model_component(
    input_data: Input[Dataset],
    output_model: Output[Model],
    metrics: Output[Metrics],
    test_size: float = 0.2,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 7
):
    """Train XGBoost model"""
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import joblib
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from datetime import datetime
    import json
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"🎯 MODEL TRAINING COMPONENT - Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv(input_data.path)
    print(f"\n📊 Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    # Split data
    features = ["PM10", "O3", "CO", "hour", "dayofweek", "month"]
    n = len(df)
    split_idx = int(n * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df["PM2.5"]
    X_test = test_df[features]
    y_test = test_df["PM2.5"]
    
    print(f"\n📈 Data Split (test_size={test_size}):")
    print(f"   Training set: {len(X_train):,} samples ({len(X_train)/n*100:.1f}%)")
    print(f"   Test set: {len(X_test):,} samples ({len(X_test)/n*100:.1f}%)")
    print(f"   Features: {len(features)}")
    print(f"\n🎯 Target Statistics:")
    print(f"   Train - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}, range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Test - mean: {y_test.mean():.2f}, std: {y_test.std():.2f}, range: [{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Train XGBoost
    print(f"\n🤖 XGBoost Model Configuration:")
    print(f"   n_estimators: {n_estimators}")
    print(f"   learning_rate: {learning_rate}")
    print(f"   max_depth: {max_depth}")
    print(f"   subsample: 0.9")
    print(f"   colsample_bytree: 0.9")
    print(f"   tree_method: hist")
    print(f"   device: cpu")
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        device="cpu",
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42
    )
    
    print(f"\n🔄 Training model...")
    train_start = datetime.now()
    model.fit(X_train, y_train, verbose=False)
    train_duration = (datetime.now() - train_start).total_seconds()
    print(f"   ✓ Training completed in {train_duration:.2f} seconds")
    
    # Evaluate
    print(f"\n📊 Evaluating model...")
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n✅ Model Performance Metrics:")
    print(f"   RMSE: {rmse:.2f} µg/m³")
    print(f"   MAE: {mae:.2f} µg/m³")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Max prediction error: {np.abs(y_test - y_pred).max():.2f} µg/m³")
    
    # Feature importance
    print(f"\n🔍 Top 3 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for idx, row in feature_importance.head(3).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    joblib.dump(model, output_model.path)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n💾 Model saved to: {output_model.path}")
    print(f"⏱️  Total duration: {duration:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Log metrics
    metrics.log_metric("rmse", rmse)
    metrics.log_metric("mae", mae)
    metrics.log_metric("r2", r2)
    metrics.log_metric("mape", mape)
    metrics.log_metric("test_samples", len(X_test))
    metrics.log_metric("train_duration_sec", train_duration)


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "numpy<2.0",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "seaborn"
    ],
    base_image="python:3.10"
)
def evaluate_model_component(
    input_data: Input[Dataset],
    input_model: Input[Model],
    metrics_output: Output[Metrics],
    test_size: float = 0.2
):
    """Evaluate model and generate plots"""
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"📊 MODEL EVALUATION COMPONENT - Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load data and model
    print(f"\n📂 Loading data and model...")
    df = pd.read_csv(input_data.path)
    model = joblib.load(input_model.path)
    print(f"   ✓ Data loaded: {len(df):,} rows")
    print(f"   ✓ Model loaded: {type(model).__name__}")
    
    # Split data (same as training)
    features = ["PM10", "O3", "CO", "hour", "dayofweek", "month"]
    n = len(df)
    split_idx = int(n * (1 - test_size))
    test_df = df.iloc[split_idx:]
    
    X_test = test_df[features]
    y_test = test_df["PM2.5"]
    
    print(f"\n🧪 Test Set: {len(X_test):,} samples")
    
    # Predict
    print(f"\n🔮 Generating predictions...")
    pred_start = datetime.now()
    y_pred = model.predict(X_test)
    pred_duration = (datetime.now() - pred_start).total_seconds()
    print(f"   ✓ Predictions completed in {pred_duration:.2f} seconds")
    print(f"   Avg prediction time: {pred_duration/len(X_test)*1000:.2f} ms per sample")
    
    # Calculate metrics
    print(f"\n📈 Computing evaluation metrics...")
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Additional metrics
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    
    print(f"\n✅ Evaluation Metrics:")
    print(f"   RMSE: {rmse:.2f} µg/m³")
    print(f"   MAE: {mae:.2f} µg/m³")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"\n📊 Error Analysis:")
    print(f"   Mean error: {errors.mean():.2f} µg/m³")
    print(f"   Std error: {errors.std():.2f} µg/m³")
    print(f"   Max error: {abs_errors.max():.2f} µg/m³")
    print(f"   90th percentile error: {np.percentile(abs_errors, 90):.2f} µg/m³")
    print(f"\n🎯 Prediction Quality:")
    within_10 = (abs_errors <= 10).sum() / len(abs_errors) * 100
    within_20 = (abs_errors <= 20).sum() / len(abs_errors) * 100
    print(f"   Within ±10 µg/m³: {within_10:.1f}%")
    print(f"   Within ±20 µg/m³: {within_20:.1f}%")
    
    # Log metrics
    metrics_output.log_metric("rmse", float(rmse))
    metrics_output.log_metric("mae", float(mae))
    metrics_output.log_metric("r2", float(r2))
    metrics_output.log_metric("mape", float(mape))
    metrics_output.log_metric("within_10_pct", float(within_10))
    metrics_output.log_metric("within_20_pct", float(within_20))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Duration: {duration:.2f} seconds")
    print(f"{'='*80}\n")


@component(
    packages_to_install=[
        "pandas>=2.0.0",
        "numpy<2.0",
        "scipy",
        "scikit-learn"
    ],
    base_image="python:3.10"
)
def drift_detection_component(
    input_data: Input[Dataset],
    metrics_output: Output[Metrics],
    test_size: float = 0.2
):
    """Detect data drift using KS test and PSI"""
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp
    from datetime import datetime
    import json
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"🔍 DRIFT DETECTION COMPONENT - Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv(input_data.path)
    print(f"\n📊 Dataset: {len(df):,} rows")
    
    # Split into reference (train) and current (test)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    
    reference_df = df.iloc[:split_idx]
    current_df = df.iloc[split_idx:]
    
    print(f"   Reference set (train): {len(reference_df):,} rows")
    print(f"   Current set (test): {len(current_df):,} rows")
    
    features = ["PM10", "O3", "CO", "hour"]
    drift_results = {}
    
    print(f"\n🔬 Performing Kolmogorov-Smirnov Tests (α=0.05):")
    print(f"   Testing {len(features)} features for distribution drift...\n")
    
    for feature in features:
        # KS Test
        ref_data = reference_df[feature].dropna()
        curr_data = current_df[feature].dropna()
        
        ks_stat, p_value = ks_2samp(ref_data, curr_data)
        
        drift_detected = p_value < 0.05
        
        drift_results[feature] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": bool(drift_detected),
            "ref_mean": float(ref_data.mean()),
            "curr_mean": float(curr_data.mean()),
            "mean_shift": float(curr_data.mean() - ref_data.mean())
        }
        
        status = "⚠️ DRIFT" if drift_detected else "✓ OK"
        print(f"   [{status}] {feature:10s}: KS={ks_stat:.4f}, p={p_value:.4f}")
        print(f"             Mean shift: {drift_results[feature]['ref_mean']:.2f} → {drift_results[feature]['curr_mean']:.2f} (Δ{drift_results[feature]['mean_shift']:+.2f})")
    
    # Log overall drift status
    total_drifted = sum(1 for r in drift_results.values() if r["drift_detected"])
    drift_percentage = (total_drifted / len(features)) * 100
    
    print(f"\n📈 Drift Summary:")
    print(f"   Features tested: {len(features)}")
    print(f"   Features with drift: {total_drifted}")
    print(f"   Drift percentage: {drift_percentage:.1f}%")
    
    if total_drifted > 0:
        print(f"\n⚠️  WARNING: Drift detected in {total_drifted} feature(s)!")
        print(f"   Consider retraining the model with recent data.")
    else:
        print(f"\n✅ No significant drift detected. Model is stable.")
    
    metrics_output.log_metric("drift_percentage", float(drift_percentage))
    metrics_output.log_metric("features_drifted", int(total_drifted))
    metrics_output.log_metric("features_tested", len(features))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Duration: {duration:.2f} seconds")
    print(f"{'='*80}\n")


@dsl.pipeline(
    name="PM2.5 Air Quality Prediction Pipeline",
    description="End-to-end MLOps pipeline for PM2.5 prediction with XGBoost"
)
def pm25_prediction_pipeline(
    data_path: str = "data/kaggle_csvs",
    test_size: float = 0.2,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 7
):
    """
    Complete ML pipeline with:
    1. Data Ingestion
    2. Data Preprocessing
    3. Model Training (XGBoost)
    4. Model Evaluation
    5. Drift Detection
    """
    
    # Step 1: Data Ingestion
    ingestion_task = data_ingestion_component(data_path=data_path)
    
    # Step 2: Data Preprocessing
    preprocessing_task = data_preprocessing_component(
        input_data=ingestion_task.outputs["output_data"],
        test_size=test_size
    )
    
    # Step 3: Model Training
    training_task = train_model_component(
        input_data=preprocessing_task.outputs["output_data"],
        test_size=test_size,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    # Step 4: Model Evaluation
    evaluation_task = evaluate_model_component(
        input_data=preprocessing_task.outputs["output_data"],
        input_model=training_task.outputs["output_model"],
        test_size=test_size
    )
    
    # Step 5: Drift Detection
    drift_task = drift_detection_component(
        input_data=preprocessing_task.outputs["output_data"],
        test_size=test_size
    )


# Compile pipeline
if __name__ == "__main__":
    from kfp import compiler
    
    pipeline_file = "pm25_pipeline.yaml"
    
    compiler.Compiler().compile(
        pipeline_func=pm25_prediction_pipeline,
        package_path=pipeline_file
    )
    
    print(f"✅ Pipeline compiled successfully: {pipeline_file}")
    print("\nTo run this pipeline:")
    print("1. Deploy to Kubeflow: kfp run submit -e <experiment> -f pm25_pipeline.yaml")
    print("2. Or use Python SDK:")
    print("   client = kfp.Client(host='<kubeflow-host>')")
    print("   client.create_run_from_pipeline_func(pm25_prediction_pipeline)")
