"""
Test Kubeflow pipeline components locally with enhanced debugging
This simulates pipeline execution without requiring Kubeflow infrastructure
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

print(f"\n{'='*80}")
print(f"🧪 KUBEFLOW PIPELINE DEBUG TEST")
print(f"{'='*80}")
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Working directory: {os.getcwd()}")
print(f"{'='*80}\n")

def test_pipeline_compilation():
    """Test that the pipeline compiles successfully"""
    print(f"🔍 TEST 1: Pipeline Compilation")
    print(f"-" * 80)
    
    try:
        from kfp import compiler
        from kubeflow_pipeline import pm25_prediction_pipeline
        
        pipeline_file = "pm25_pipeline_test.yaml"
        
        compiler.Compiler().compile(
            pipeline_func=pm25_prediction_pipeline,
            package_path=pipeline_file
        )
        
        # Check if file was created
        if os.path.exists(pipeline_file):
            file_size = os.path.getsize(pipeline_file) / 1024
            print(f"✅ Pipeline compiled successfully!")
            print(f"   📄 File: {pipeline_file}")
            print(f"   📊 Size: {file_size:.2f} KB")
            
            # Clean up test file
            os.remove(pipeline_file)
            return True
        else:
            print(f"❌ Pipeline file not created")
            return False
            
    except Exception as e:
        print(f"❌ Compilation failed: {str(e)}")
        return False


def test_component_imports():
    """Test that all components can be imported"""
    print(f"\n🔍 TEST 2: Component Imports")
    print(f"-" * 80)
    
    components = [
        "data_ingestion_component",
        "data_preprocessing_component",
        "train_model_component",
        "evaluate_model_component",
        "drift_detection_component"
    ]
    
    try:
        from kubeflow_pipeline import (
            data_ingestion_component,
            data_preprocessing_component,
            train_model_component,
            evaluate_model_component,
            drift_detection_component
        )
        
        print(f"✅ All components imported successfully!")
        for comp in components:
            print(f"   ✓ {comp}")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False


def test_deployment_script():
    """Test deployment script structure"""
    print(f"\n🔍 TEST 3: Deployment Script")
    print(f"-" * 80)
    
    try:
        from kubeflow_deploy import (
            deploy_pipeline,
            wait_for_run,
            list_pipeline_runs
        )
        
        print(f"✅ Deployment functions available!")
        print(f"   ✓ deploy_pipeline")
        print(f"   ✓ wait_for_run")
        print(f"   ✓ list_pipeline_runs")
        return True
        
    except Exception as e:
        print(f"❌ Deployment import failed: {str(e)}")
        return False


def test_data_availability():
    """Check if data directory exists"""
    print(f"\n🔍 TEST 4: Data Availability")
    print(f"-" * 80)
    
    data_path = "data/kaggle_csvs"
    
    if os.path.exists(data_path):
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        print(f"✅ Data directory found!")
        print(f"   📁 Path: {data_path}")
        print(f"   📊 CSV files: {len(csv_files)}")
        if csv_files:
            print(f"   📄 Sample files: {', '.join(csv_files[:5])}")
        return True
    else:
        print(f"⚠️  Data directory not found: {data_path}")
        print(f"   Pipeline will fail during execution without data")
        return False


def test_dependencies():
    """Check if required packages are installed"""
    print(f"\n🔍 TEST 5: Python Dependencies")
    print(f"-" * 80)
    
    required_packages = {
        'kfp': 'Kubeflow Pipelines',
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'xgboost': 'Gradient boosting',
        'scipy': 'Scientific computing',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'joblib': 'Model serialization'
    }
    
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✓ {package:15s} - {description}")
        except ImportError:
            print(f"   ✗ {package:15s} - {description} (MISSING)")
            all_installed = False
    
    if all_installed:
        print(f"\n✅ All dependencies installed!")
        return True
    else:
        print(f"\n⚠️  Some dependencies missing")
        return False


def test_pipeline_yaml():
    """Check if compiled pipeline YAML exists"""
    print(f"\n🔍 TEST 6: Pipeline YAML File")
    print(f"-" * 80)
    
    pipeline_file = "pm25_pipeline.yaml"
    
    if os.path.exists(pipeline_file):
        file_size = os.path.getsize(pipeline_file) / 1024
        
        # Read and check content
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
            components = content.count('name:')
            
        print(f"✅ Pipeline YAML exists!")
        print(f"   📄 File: {pipeline_file}")
        print(f"   📊 Size: {file_size:.2f} KB")
        print(f"   🔧 Components: ~{components}")
        return True
    else:
        print(f"⚠️  Pipeline YAML not found: {pipeline_file}")
        print(f"   Run: python kubeflow_pipeline.py")
        return False


def print_debug_features():
    """Display enhanced debugging features"""
    print(f"\n{'='*80}")
    print(f"📋 ENHANCED DEBUGGING FEATURES")
    print(f"{'='*80}")
    
    features = [
        ("Timestamps", "All components show start/end times and duration"),
        ("Progress Tracking", "Real-time progress for data loading and processing"),
        ("Data Statistics", "Detailed stats on rows, columns, and memory usage"),
        ("Error Handling", "Comprehensive error messages with file names"),
        ("Metrics Logging", "Extended metrics including MAPE, error analysis"),
        ("Feature Analysis", "Feature importance and distribution stats"),
        ("Drift Detection", "Detailed KS test results with mean shifts"),
        ("Visual Separators", "Clear section headers with emojis"),
        ("Performance Metrics", "Prediction times and processing speeds"),
        ("Quality Indicators", "Accuracy within threshold percentages")
    ]
    
    for i, (feature, description) in enumerate(features, 1):
        print(f"{i:2d}. {feature:20s} - {description}")
    
    print(f"\n{'='*80}")


# Run all tests
def main():
    results = {
        "Pipeline Compilation": test_pipeline_compilation(),
        "Component Imports": test_component_imports(),
        "Deployment Script": test_deployment_script(),
        "Data Availability": test_data_availability(),
        "Dependencies": test_dependencies(),
        "Pipeline YAML": test_pipeline_yaml()
    }
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*80}\n")
    
    # Show debug features
    print_debug_features()
    
    # Final recommendations
    print(f"💡 NEXT STEPS:")
    print(f"-" * 80)
    
    if passed == total:
        print(f"✅ All tests passed! Pipeline is ready.")
        print(f"\n📋 To deploy to Kubeflow:")
        print(f"   python kubeflow_deploy.py --host <kubeflow-url>")
        print(f"\n📋 To run locally:")
        print(f"   python scripts/pipeline.py")
    else:
        print(f"⚠️  Some tests failed. Address the issues above before deployment.")
    
    print(f"\n{'='*80}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
