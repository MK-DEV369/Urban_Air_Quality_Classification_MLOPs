#!/usr/bin/env python3
"""
Data ingestion module - Extract and load raw data
"""

import os
import pandas as pd
from pathlib import Path


def load_kaggle_csvs(kaggle_dir="data/kaggle_csvs"):
    """Load all CSV files from kaggle directory."""
    print(f"📂 Loading CSVs from {kaggle_dir}...")
    
    csv_files = list(Path(kaggle_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {kaggle_dir}")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error loading {csv_file}: {e}")
    
    print(f"✅ Loaded {len(dfs)} files")
    return pd.concat(dfs, ignore_index=True)


def save_raw_data(df, output_path="data/raw_combined.csv"):
    """Save combined raw data."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved raw data to {output_path}")


def main():
    print("=" * 60)
    print("DATA INGESTION PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_kaggle_csvs()
    print(f"📊 Total records: {len(df)}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Save
    save_raw_data(df)
    
    print("\n✅ Data ingestion complete!")


if __name__ == "__main__":
    main()
