#!/usr/bin/env python3
"""
Data preprocessing module - Clean and transform data
"""

import pandas as pd
import numpy as np


def preprocess_data(input_path="data/raw_combined.csv", output_path="data/master_airquality_clean.csv"):
    """Clean and preprocess air quality data."""
    print("🧹 Starting data preprocessing...")
    
    # Load
    df = pd.read_csv(input_path, low_memory=False)
    initial_count = len(df)
    print(f"📊 Initial records: {initial_count}")
    
    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    print(f"✅ Parsed timestamps: {len(df)} records")
    
    # Extract temporal features
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    df["year"] = df["Timestamp"].dt.year
    
    # Convert pollutants to numeric
    pollutants = ["PM2.5", "PM10", "O3", "CO", "NO2", "SO2"]
    for col in pollutants:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing target variable
    df.dropna(subset=["PM2.5"], inplace=True)
    print(f"✅ After dropping missing PM2.5: {len(df)} records")
    
    # Fill missing features with median
    feature_cols = ["PM10", "O3", "CO", "hour", "dayofweek", "month"]
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Remove outliers (optional - simple IQR method for PM2.5)
    Q1 = df["PM2.5"].quantile(0.01)
    Q3 = df["PM2.5"].quantile(0.99)
    df = df[(df["PM2.5"] >= Q1) & (df["PM2.5"] <= Q3)]
    print(f"✅ After outlier removal: {len(df)} records")
    
    # Sort by timestamp
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"💾 Saved clean data to {output_path}")
    
    print(f"\n📊 Preprocessing Summary:")
    print(f"   Initial: {initial_count:,}")
    print(f"   Final: {len(df):,}")
    print(f"   Reduction: {(1 - len(df)/initial_count)*100:.1f}%")


def main():
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    preprocess_data()
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
