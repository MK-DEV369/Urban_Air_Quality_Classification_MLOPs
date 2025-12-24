#!/usr/bin/env python3
"""
Complete ETL -> Training -> Evaluation Pipeline
"""

import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

import data_ingestion
import data_preprocessing
import train_with_comet
import evaluate_model
import shap_analysis


def run_pipeline(skip_ingestion=False):
    """Run complete ML pipeline."""
    
    print("\n" + "="*70)
    print(" "*20 + "ML PIPELINE ORCHESTRATION")
    print("="*70 + "\n")
    
    try:
        # Step 1: Data Ingestion
        if not skip_ingestion:
            print("\n[STEP 1/5] DATA INGESTION")
            print("-" * 70)
            data_ingestion.main()
        else:
            print("\n[STEP 1/5] DATA INGESTION - SKIPPED")
        
        # Step 2: Data Preprocessing
        print("\n[STEP 2/5] DATA PREPROCESSING")
        print("-" * 70)
        data_preprocessing.main()
        
        # Step 3: Model Training
        print("\n[STEP 3/5] MODEL TRAINING")
        print("-" * 70)
        train_with_comet.main()
        
        # Step 4: Model Evaluation
        print("\n[STEP 4/5] MODEL EVALUATION")
        print("-" * 70)
        evaluate_model.main()
        
        # Step 5: Model Interpretability
        print("\n[STEP 5/5] MODEL INTERPRETABILITY (SHAP)")
        print("-" * 70)
        shap_analysis.main()
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Skip ingestion if raw data already exists
    skip = os.path.exists("data/raw_combined.csv")
    run_pipeline(skip_ingestion=skip)
