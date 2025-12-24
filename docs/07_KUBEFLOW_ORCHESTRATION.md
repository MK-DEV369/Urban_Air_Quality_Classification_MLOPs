# Kubeflow Pipeline Orchestration

## Overview

This project implements pipeline orchestration using **Kubeflow Pipelines**, a platform for building and deploying portable, scalable machine learning workflows based on Docker containers.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Kubeflow Pipelines Platform                │
│  (Running on Kubernetes cluster or local deployment)   │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼─────┐            ┌──────▼──────┐
    │ Pipeline │            │  Pipeline   │
    │   DAG    │            │ Components  │
    └────┬─────┘            └──────┬──────┘
         │                         │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │                         │
    ┌────▼─────┐  ┌────▼─────┐  ┌─▼────┐
    │Container │  │Container │  │ ...  │
    │  Step 1  │  │  Step 2  │  │      │
    └──────────┘  └──────────┘  └──────┘
```

---

## Pipeline Components

### 1. Data Ingestion Component

**Purpose**: Load and combine raw CSV files

**Inputs**: Data path (`data/kaggle_csvs`)

**Outputs**: Combined dataset (CSV)

**Container**: Python 3.10 with pandas, numpy, pyarrow

**Code**: `kubeflow_pipeline.py` → `data_ingestion_component()`

---

### 2. Data Preprocessing Component

**Purpose**: Clean, transform, and feature engineering

**Inputs**: Raw combined dataset

**Outputs**: Cleaned dataset with temporal features

**Processing**:
- Parse timestamps
- Extract hour, dayofweek, month
- Handle missing values
- Feature selection

**Container**: Python 3.10 with pandas, numpy, scikit-learn

**Code**: `kubeflow_pipeline.py` → `data_preprocessing_component()`

---

### 3. Model Training Component

**Purpose**: Train XGBoost regression model

**Inputs**: Preprocessed dataset

**Outputs**: 
- Trained model (joblib)
- Training metrics (RMSE, MAE, R²)

**Hyperparameters**:
- `n_estimators`: 300
- `learning_rate`: 0.05
- `max_depth`: 7
- `subsample`: 0.9

**Container**: Python 3.10 with xgboost, scikit-learn, joblib

**Code**: `kubeflow_pipeline.py` → `train_model_component()`

---

### 4. Model Evaluation Component

**Purpose**: Evaluate model performance and generate metrics

**Inputs**: 
- Preprocessed dataset
- Trained model

**Outputs**: Evaluation metrics (RMSE, MAE, R², MAPE)

**Container**: Python 3.10 with scikit-learn, matplotlib, seaborn

**Code**: `kubeflow_pipeline.py` → `evaluate_model_component()`

---

### 5. Drift Detection Component

**Purpose**: Detect data drift using statistical tests

**Inputs**: Preprocessed dataset

**Outputs**: Drift metrics (KS statistic, p-value, drift percentage)

**Tests**:
- Kolmogorov-Smirnov test
- Population Stability Index (PSI)

**Container**: Python 3.10 with scipy, pandas

**Code**: `kubeflow_pipeline.py` → `drift_detection_component()`

---

## Pipeline DAG

```
Data Ingestion
      │
      ▼
Data Preprocessing
      │
      ├─────────────┬─────────────┐
      │             │             │
      ▼             ▼             ▼
   Training    Evaluation    Drift Detection
      │             │             │
      └─────────────┴─────────────┘
               (Complete)
```

---

## Installation

### Prerequisites

1. **Kubernetes Cluster** (optional for local development)
2. **Kubeflow Pipelines** installed
3. **Python 3.10+** with pip

### Install Kubeflow SDK

```bash
pip install kfp==2.7.0
```

### Verify Installation

```bash
python -c "import kfp; print(kfp.__version__)"
```

---

## Compiling the Pipeline

### Method 1: Using Python Script

```bash
python kubeflow_pipeline.py
```

**Output**: `pm25_pipeline.yaml`

### Method 2: Using KFP CLI

```bash
kfp pipeline compile \
  --py kubeflow_pipeline.py \
  --output pm25_pipeline.yaml
```

---

## Deploying the Pipeline

### Local Deployment (Standalone)

For local testing without full Kubeflow installation:

```bash
# Install standalone Kubeflow Pipelines
docker run -p 8080:8080 gcr.io/ml-pipeline/frontend:2.0.0
```

### Deploy to Kubeflow

#### Option 1: Using Python SDK

```python
import kfp

client = kfp.Client(host="http://localhost:8080")

client.create_run_from_pipeline_func(
    pipeline_func=pm25_prediction_pipeline,
    experiment_name="pm25-airquality-exp",
    run_name="pipeline-run-1"
)
```

#### Option 2: Using Deployment Script

```bash
python kubeflow_deploy.py \
  --host http://localhost:8080 \
  --experiment pm25-airquality-exp \
  --run-name pipeline-run-1
```

#### Option 3: Using Web UI

1. Navigate to Kubeflow Pipelines UI (http://localhost:8080)
2. Click **Upload Pipeline**
3. Select `pm25_pipeline.yaml`
4. Create a new run
5. Configure parameters:
   - `data_path`: `data/kaggle_csvs`
   - `test_size`: `0.2`
   - `n_estimators`: `300`
   - `learning_rate`: `0.05`
   - `max_depth`: `7`
6. Click **Start**

---

## Running the Pipeline

### Automated Deployment Script

```bash
# Run and wait for completion
python kubeflow_deploy.py --wait

# List all runs
python kubeflow_deploy.py --list

# Custom parameters
python kubeflow_deploy.py \
  --host http://kubeflow.example.com \
  --experiment my-experiment \
  --run-name test-run-1
```

### Manual Execution

```python
from kubeflow_pipeline import pm25_prediction_pipeline
import kfp

client = kfp.Client(host="http://localhost:8080")

run = client.create_run_from_pipeline_func(
    pipeline_func=pm25_prediction_pipeline,
    arguments={
        "data_path": "data/kaggle_csvs",
        "test_size": 0.2,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 7
    }
)

print(f"Run ID: {run.run_id}")
```

---

## Monitoring Pipeline Execution

### View in UI

1. Go to Kubeflow UI (http://localhost:8080)
2. Click **Experiments**
3. Select your experiment
4. Click on the run name
5. View DAG, logs, and metrics

### Programmatic Monitoring

```python
import kfp

client = kfp.Client(host="http://localhost:8080")

# Wait for completion
run = client.wait_for_run_completion(run_id, timeout=3600)

# Check status
print(f"Status: {run.run.status}")
```

### Access Logs

```python
# Get component logs
client.get_run(run_id)
```

---

## Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | `data/kaggle_csvs` | Path to raw CSV files |
| `test_size` | float | `0.2` | Test split ratio (0-1) |
| `n_estimators` | int | `300` | Number of boosting rounds |
| `learning_rate` | float | `0.05` | XGBoost learning rate |
| `max_depth` | int | `7` | Maximum tree depth |

---

## Artifacts and Outputs

### Kubeflow Artifacts

All outputs are stored in MinIO (Kubeflow's artifact store):

| Artifact | Type | Location |
|----------|------|----------|
| Ingested Data | Dataset | `minio://mlpipeline/<run-id>/ingestion/output_data` |
| Preprocessed Data | Dataset | `minio://mlpipeline/<run-id>/preprocessing/output_data` |
| Trained Model | Model | `minio://mlpipeline/<run-id>/training/output_model` |
| Metrics | JSON | `minio://mlpipeline/<run-id>/{component}/metrics` |

### Accessing Artifacts

```python
# Download model artifact
client = kfp.Client(host="http://localhost:8080")

artifact_path = client.get_artifact(
    run_id=run_id,
    node_id="training",
    artifact_name="output_model"
)

print(f"Model downloaded to: {artifact_path}")
```

---

## Scheduling Pipelines

### Recurring Runs

Create scheduled pipeline runs:

```python
from kfp import Client
from datetime import datetime

client = Client(host="http://localhost:8080")

# Create recurring run (monthly)
job = client.create_recurring_run(
    experiment_id=experiment.id,
    job_name="pm25-monthly-training",
    description="Monthly model retraining",
    pipeline_package_path="pm25_pipeline.yaml",
    params={
        "data_path": "data/kaggle_csvs",
        "test_size": 0.2
    },
    cron_expression="0 0 1 * *",  # 1st of every month at midnight
    max_concurrency=1,
    enable_caching=True
)

print(f"Recurring job created: {job.id}")
```

### Cron Expressions

| Pattern | Frequency |
|---------|-----------|
| `0 0 * * *` | Daily at midnight |
| `0 0 * * 0` | Weekly (Sunday) |
| `0 0 1 * *` | Monthly (1st day) |
| `0 */6 * * *` | Every 6 hours |

---

## Caching and Optimization

### Enable Caching

```python
# Cache component outputs to avoid recomputation
@dsl.pipeline(name="cached-pipeline")
def cached_pipeline():
    ingestion_task = data_ingestion_component()
    
    # Enable caching (default: True)
    ingestion_task.set_caching_options(enable_caching=True)
```

### Benefits:
- Skip unchanged components
- Faster iterations during development
- Cost savings on cloud platforms

---

## Integration with Existing Tools

### Comet ML Integration

Add Comet ML logging to components:

```python
@component(packages_to_install=["comet-ml"])
def train_with_comet_component(...):
    import comet_ml
    
    experiment = comet_ml.Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name="pm25-kubeflow"
    )
    
    # Training code...
    experiment.log_metric("rmse", rmse)
    experiment.end()
```

### Model Registry Integration

```python
@component
def register_model_component(input_model: Input[Model]):
    from scripts.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    version_id = registry.register_model(
        model_path=input_model.path,
        model_name="pm25_predictor",
        stage="staging"
    )
```

---

## Troubleshooting

### Issue: Pipeline fails to compile

**Error**: `ModuleNotFoundError: No module named 'kfp'`

**Solution**:
```bash
pip install kfp==2.7.0
```

### Issue: Cannot connect to Kubeflow

**Error**: `Connection refused`

**Solution**:
- Check Kubeflow is running: `kubectl get pods -n kubeflow`
- Port forward: `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`

### Issue: Component execution fails

**Solution**:
- Check component logs in Kubeflow UI
- Verify base image has all dependencies
- Add debug print statements in component code

### Issue: Data not found

**Solution**:
- Ensure data is accessible to Kubernetes pods
- Mount volumes or use cloud storage (S3, GCS)
- Update `data_path` parameter

---

## Kubeflow vs. GitHub Actions

| Feature | Kubeflow | GitHub Actions |
|---------|----------|----------------|
| **Use Case** | Complex ML workflows | CI/CD automation |
| **Execution** | Kubernetes cluster | GitHub-hosted runners |
| **Scalability** | High (distributed) | Medium (single runner) |
| **Artifact Management** | MinIO built-in | Manual upload/download |
| **UI** | Rich DAG visualization | Simple logs view |
| **Caching** | Smart component caching | Basic caching |
| **Cost** | Requires K8s cluster | Free tier available |

**Recommendation**: 
- Use **Kubeflow** for production ML workflows at scale
- Use **GitHub Actions** for lightweight CI/CD and testing

---

## Production Deployment

### Deploy on Cloud

#### Google Cloud (GKE + Kubeflow)

```bash
# Create GKE cluster
gcloud container clusters create ml-cluster \
  --machine-type n1-standard-4 \
  --num-nodes 3

# Install Kubeflow
kfctl apply -f kfctl_gcp_iap.yaml
```

#### AWS (EKS + Kubeflow)

```bash
# Create EKS cluster
eksctl create cluster --name ml-cluster --region us-west-2

# Install Kubeflow
kfctl apply -f kfctl_aws.yaml
```

### Authentication

Set up authentication for remote clusters:

```python
import kfp

client = kfp.Client(
    host="https://kubeflow.example.com",
    existing_token="your-auth-token"
)
```

---

## Best Practices

1. **Modular Components**: Keep components small and focused
2. **Parameterization**: Use pipeline parameters for flexibility
3. **Caching**: Enable caching for unchanged components
4. **Error Handling**: Add try-except in components for robustness
5. **Logging**: Use print statements for debugging
6. **Versioning**: Tag pipeline YAML files with versions
7. **Testing**: Test components individually before full pipeline
8. **Resource Limits**: Set CPU/memory limits for components

---

## References

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [KFP Python SDK](https://kubeflow-pipelines.readthedocs.io/)
- [Kubeflow Examples](https://github.com/kubeflow/pipelines/tree/master/samples)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
