#!/usr/bin/env python3
"""
Model versioning and registry management
Lightweight alternative to MLflow model registry
"""

import os
import json
import shutil
import joblib
from datetime import datetime
from pathlib import Path


class ModelRegistry:
    """Simple file-based model registry."""
    
    def __init__(self, registry_path="models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_path / "registry_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"models": {}, "versions": []}
    
    def _save_index(self):
        """Save registry index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=4)
    
    def register_model(self, model_path, model_name, metrics, stage="staging", tags=None):
        """Register a new model version."""
        
        # Generate version ID
        version = len(self.index["versions"]) + 1
        version_id = f"v{version}"
        timestamp = datetime.now().isoformat()
        
        # Create version directory
        version_dir = self.registry_path / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_filename = Path(model_path).name
        dest_path = version_dir / model_filename
        shutil.copy2(model_path, dest_path)
        
        # Save metadata
        metadata = {
            "version": version_id,
            "model_name": model_name,
            "registered_at": timestamp,
            "stage": stage,
            "metrics": metrics,
            "tags": tags or [],
            "model_file": str(dest_path)
        }
        
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update index
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = []
        
        self.index["models"][model_name].append(version_id)
        self.index["versions"].append({
            "version_id": version_id,
            "model_name": model_name,
            "stage": stage,
            "registered_at": timestamp
        })
        
        self._save_index()
        
        print(f"✅ Registered {model_name} {version_id} with stage '{stage}'")
        return version_id
    
    def promote_model(self, model_name, version_id, stage):
        """Promote model to a different stage (staging -> production)."""
        version_dir = self.registry_path / model_name / version_id
        metadata_file = version_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise ValueError(f"Model {model_name} {version_id} not found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata["stage"] = stage
        metadata["promoted_at"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update index
        for v in self.index["versions"]:
            if v["version_id"] == version_id and v["model_name"] == model_name:
                v["stage"] = stage
        
        self._save_index()
        
        print(f"✅ Promoted {model_name} {version_id} to {stage}")
    
    def get_model(self, model_name, version_id=None, stage="production"):
        """Load a specific model version or latest from stage."""
        
        if version_id:
            version_dir = self.registry_path / model_name / version_id
        else:
            # Get latest version in specified stage
            candidates = []
            for v in self.index["versions"]:
                if v["model_name"] == model_name and v["stage"] == stage:
                    candidates.append(v)
            
            if not candidates:
                raise ValueError(f"No {stage} version found for {model_name}")
            
            # Sort by timestamp and get latest
            candidates.sort(key=lambda x: x["registered_at"], reverse=True)
            version_id = candidates[0]["version_id"]
            version_dir = self.registry_path / model_name / version_id
        
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        model = joblib.load(metadata["model_file"])
        
        return model, metadata
    
    def list_versions(self, model_name=None):
        """List all registered model versions."""
        if model_name:
            return [v for v in self.index["versions"] if v["model_name"] == model_name]
        return self.index["versions"]


def main():
    """Demo usage of model registry."""
    
    registry = ModelRegistry()
    
    # Check if model exists
    model_path = "models/best_pm25_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Register model
    metrics = {"rmse": 25.5, "mae": 18.3, "r2": 0.82}
    version_id = registry.register_model(
        model_path=model_path,
        model_name="pm25_predictor",
        metrics=metrics,
        stage="staging",
        tags=["xgboost", "production-candidate"]
    )
    
    # List versions
    print("\n📋 Registered versions:")
    versions = registry.list_versions("pm25_predictor")
    for v in versions:
        print(f"   {v['version_id']} - Stage: {v['stage']} - {v['registered_at']}")
    
    # Promote to production
    print(f"\n🚀 Promoting {version_id} to production...")
    registry.promote_model("pm25_predictor", version_id, "production")
    
    # Load production model
    print("\n📦 Loading production model...")
    model, metadata = registry.get_model("pm25_predictor", stage="production")
    print(f"   Loaded: {metadata['model_name']} {metadata['version']}")
    print(f"   Metrics: {metadata['metrics']}")


if __name__ == "__main__":
    main()
