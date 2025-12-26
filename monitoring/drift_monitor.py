#!/usr/bin/env python3
"""
Data drift monitoring using custom implementation
Alternative to Evidently AI
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime


def calculate_ks_statistic(reference_data, current_data, feature):
    """Calculate Kolmogorov-Smirnov statistic for drift detection."""
    if feature not in reference_data.columns or feature not in current_data.columns:
        return None
    
    ref_values = pd.to_numeric(reference_data[feature], errors='coerce').dropna()
    curr_values = pd.to_numeric(current_data[feature], errors='coerce').dropna()
    
    if len(ref_values) < 10 or len(curr_values) < 10:
        return None
    
    statistic, p_value = stats.ks_2samp(ref_values, curr_values)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05)
    }


def calculate_psi(reference_data, current_data, feature, bins=10):
    """Calculate Population Stability Index (PSI) for drift detection."""
    if feature not in reference_data.columns or feature not in current_data.columns:
        return None
    
    ref_values = pd.to_numeric(reference_data[feature], errors='coerce').dropna()
    curr_values = pd.to_numeric(current_data[feature], errors='coerce').dropna()
    
    if len(ref_values) < 10 or len(curr_values) < 10:
        return None
    
    # Create bins based on reference data
    _, bin_edges = np.histogram(ref_values, bins=bins)
    
    # Calculate distributions
    ref_dist, _ = np.histogram(ref_values, bins=bin_edges)
    curr_dist, _ = np.histogram(curr_values, bins=bin_edges)
    
    # Normalize
    ref_dist = (ref_dist + 1) / (len(ref_values) + bins)
    curr_dist = (curr_dist + 1) / (len(curr_values) + bins)
    
    # Calculate PSI
    psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
    
    # PSI interpretation: <0.1 = no drift, 0.1-0.2 = moderate, >0.2 = significant
    drift_level = "none" if psi < 0.1 else ("moderate" if psi < 0.2 else "significant")
    
    return {
        "psi": float(psi),
        "drift_level": drift_level,
        "drift_detected": bool(psi > 0.1)
    }


def detect_drift(reference_csv="data/master_airquality_clean.csv", 
                 current_csv="data/master_airquality_clean.csv",
                 output_dir="monitoring/reports"):
    """Detect data drift between reference and current datasets."""
    
    print("🔍 Starting drift detection...")
    
    # Load data
    ref_df = pd.read_csv(reference_csv, low_memory=False)
    curr_df = pd.read_csv(current_csv, low_memory=False)
    
    # Use only recent data for current (simulate production data)
    # Take last 20% as "current" if using same file
    split_point = int(len(curr_df) * 0.8)
    ref_df = ref_df.iloc[:split_point]
    curr_df = curr_df.iloc[split_point:]
    
    print(f"📊 Reference data: {len(ref_df)} samples")
    print(f"📊 Current data: {len(curr_df)} samples")
    
    # Features to monitor
    features = ["PM10", "O3", "CO", "PM2.5"]
    
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "reference_size": len(ref_df),
        "current_size": len(curr_df),
        "features": {}
    }
    
    # Calculate drift metrics for each feature
    for feature in features:
        print(f"   Analyzing {feature}...")
        
        ks_result = calculate_ks_statistic(ref_df, curr_df, feature)
        psi_result = calculate_psi(ref_df, curr_df, feature)
        
        if ks_result and psi_result:
            drift_report["features"][feature] = {
                "ks_test": ks_result,
                "psi": psi_result,
                "drift_detected": bool(ks_result["drift_detected"] or psi_result["drift_detected"])
            }
            
            if drift_report["features"][feature]["drift_detected"]:
                print(f"      ⚠️  DRIFT DETECTED for {feature}")
            else:
                print(f"      ✅ No drift for {feature}")
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    drift_report_serializable = convert_to_json_serializable(drift_report)
    
    with open(report_file, 'w') as f:
        json.dump(drift_report_serializable, f, indent=4)
    
    print(f"\n💾 Drift report saved to {report_file}")
    
    # Summary
    total_features = len(features)
    drifted_features = sum(1 for f in drift_report["features"].values() if f["drift_detected"])
    
    print(f"\n📊 Drift Summary:")
    print(f"   Features monitored: {total_features}")
    print(f"   Features with drift: {drifted_features}")
    print(f"   Drift percentage: {drifted_features/total_features*100:.1f}%")
    
    return drift_report


def main():
    print("=" * 60)
    print("DATA DRIFT MONITORING")
    print("=" * 60 + "\n")
    
    detect_drift()
    
    print("\n✅ Drift monitoring complete!")


if __name__ == "__main__":
    main()
