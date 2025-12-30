#!/usr/bin/env python3
"""
Data drift monitoring using custom implementation
Alternative to Evidently AI

Uses ensemble approach with KS test + PSI for robust drift detection
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def calculate_ks_statistic(reference_data, current_data, feature, threshold=0.10):
    """
    Calculate Kolmogorov-Smirnov statistic for drift detection.
    
    Note: KS test is very sensitive to large sample sizes.
    Use threshold on statistic value (not p-value) for practical results.
    """
    if feature not in reference_data.columns or feature not in current_data.columns:
        return None
    
    ref_values = pd.to_numeric(reference_data[feature], errors='coerce').dropna()
    curr_values = pd.to_numeric(current_data[feature], errors='coerce').dropna()
    
    if len(ref_values) < 10 or len(curr_values) < 10:
        return None
    
    # Sample if data is too large (reduces sensitivity to sample size)
    sample_size = min(10000, len(ref_values), len(curr_values))
    ref_sampled = np.random.choice(ref_values, size=sample_size, replace=False)
    curr_sampled = np.random.choice(curr_values, size=sample_size, replace=False)
    
    statistic, p_value = stats.ks_2samp(ref_sampled, curr_sampled)
    
    # Use statistic threshold instead of p-value for practical decision
    # KS statistic > 0.1 indicates meaningful difference
    practical_drift = bool(statistic > threshold)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "threshold": threshold,
        "drift_detected": practical_drift,
        "interpretation": "meaningful" if practical_drift else "negligible"
    }


def calculate_psi(reference_data, current_data, feature, bins=10):
    """
    Calculate Population Stability Index (PSI) for drift detection.
    PSI is more stable and practical than KS for large datasets.
    """
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
    
    # Normalize with Laplace smoothing to avoid log(0)
    ref_dist = (ref_dist + 1) / (len(ref_values) + bins)
    curr_dist = (curr_dist + 1) / (len(curr_values) + bins)
    
    # Calculate PSI
    psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
    
    # PSI interpretation: 
    # <0.1 = none, 0.1-0.25 = small, 0.25-1.0 = moderate, >1.0 = significant
    if psi < 0.1:
        drift_level = "none"
    elif psi < 0.25:
        drift_level = "small"
    elif psi < 1.0:
        drift_level = "moderate"
    else:
        drift_level = "significant"
    
    return {
        "psi": float(psi),
        "drift_level": drift_level,
        "drift_detected": bool(psi > 0.25),  # Use 0.25 as practical threshold
        "severity": 1 if psi > 1.0 else (2 if psi > 0.25 else (3 if psi > 0.1 else 4))
    }


def detect_drift(reference_csv="data/master_airquality_clean.csv", 
                 current_csv="data/master_airquality_clean.csv",
                 output_dir="monitoring/reports",
                 ensemble_voting=True):
    """
    Detect data drift between reference and current datasets.
    
    Args:
        reference_csv: Path to reference (baseline) data
        current_csv: Path to current (production) data
        output_dir: Directory for drift reports
        ensemble_voting: Require agreement between KS and PSI for drift detection
    """
    
    print("🔍 Starting drift detection (ensemble approach)...")
    
    # Load data
    ref_df = pd.read_csv(reference_csv, low_memory=False)
    curr_df = pd.read_csv(current_csv, low_memory=False)
    
    # Use 80/20 split: first 80% as reference, last 20% as current
    split_point = int(len(curr_df) * 0.8)
    ref_df = ref_df.iloc[:split_point].reset_index(drop=True)
    curr_df = curr_df.iloc[split_point:].reset_index(drop=True)
    
    print(f"📊 Reference data: {len(ref_df):,} samples (baseline)")
    print(f"📊 Current data: {len(curr_df):,} samples (production)")
    print(f"⚙️  Using ensemble voting (KS + PSI agreement required)\n")
    
    # Features to monitor
    features = ["PM10", "O3", "CO", "PM2.5"]
    
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "reference_size": len(ref_df),
        "current_size": len(curr_df),
        "ensemble_method": "voting (KS + PSI)",
        "features": {}
    }
    
    # Calculate drift metrics for each feature
    for feature in features:
        print(f"   Analyzing {feature}...")
        
        ks_result = calculate_ks_statistic(ref_df, curr_df, feature)
        psi_result = calculate_psi(ref_df, curr_df, feature)
        
        if ks_result and psi_result:
            # Ensemble voting: require agreement between KS and PSI
            # Both methods must agree drift exists for actual drift alert
            if ensemble_voting:
                drift_detected = ks_result["drift_detected"] and psi_result["drift_detected"]
            else:
                drift_detected = ks_result["drift_detected"] or psi_result["drift_detected"]
            
            drift_report["features"][feature] = {
                "ks_test": ks_result,
                "psi": psi_result,
                "drift_detected": drift_detected,
                "consensus": "both" if (ks_result["drift_detected"] and psi_result["drift_detected"]) else "disagreement"
            }
            
            # Print results with color coding
            status = "⚠️  DRIFT" if drift_detected else "✅ NO DRIFT"
            confidence = "STRONG" if drift_report["features"][feature]["consensus"] == "both" else "WEAK/MIXED"
            print(f"      {status} (confidence: {confidence})")
            print(f"         KS: {ks_result['interpretation']} (stat={ks_result['statistic']:.4f})")
            print(f"         PSI: {psi_result['drift_level']} (psi={psi_result['psi']:.4f})")
    
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
    strong_consensus = sum(1 for f in drift_report["features"].values() if f["consensus"] == "both")
    
    print(f"\n📊 Drift Summary:")
    print(f"   Features monitored: {total_features}")
    print(f"   Features with drift (ensemble): {drifted_features}")
    print(f"   Strong consensus (both methods): {strong_consensus}")
    print(f"   Drift percentage: {drifted_features/total_features*100:.1f}%")
    
    if drifted_features == 0:
        print(f"\n   ✅ No data drift detected - model remains stable")
    else:
        print(f"\n   ⚠️  Drift detected - consider retraining model")
    
    return drift_report


def main():
    print("=" * 70)
    print("DATA DRIFT MONITORING - ENSEMBLE APPROACH (KS + PSI)")
    print("=" * 70 + "\n")
    
    detect_drift()
    
    print("\n" + "=" * 70)
    print("✅ Drift monitoring complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
