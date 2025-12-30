# Complete Execution Guide - MLOps Project Testing & Validation

This guide provides step-by-step commands to test and validate every component of the PM2.5 MLOps project.

---

## Prerequisites

```powershell
# 1. Activate virtual environment
& "E:\5th SEM Data\AI254TA-Machine Learning Operations(MLOps)\MLOPs_Project\venv\Scripts\Activate.ps1"

# 2. Verify Python and dependencies
python --version
pip list

# 3. Install/update requirements
pip install -r requirements.txt

# 4. Set environment variables (if using Comet ML / W&B)
$env:COMET_API_KEY = "your-comet-api-key"
$env:COMET_PROJECT_NAME = "pm25-mlops"
$env:WANDB_API_KEY = "your-wandb-api-key"
```

---

## 1. Data Pipeline Testing

### Data Ingestion
```powershell
# Ingest raw data (combines station CSVs)
python scripts/data_ingestion.py

# Expected output: data/raw_combined.csv, data/stations_combined.csv, data/cities_combined.csv
# Verify files exist
Test-Path data/raw_combined.csv
Test-Path data/stations_combined.csv
```

### Data Preprocessing
```powershell
# Clean and preprocess data
python scripts/data_preprocessing.py

# Expected output: data/master_airquality_clean.csv
# Verify file exists and check row count
Test-Path data/master_airquality_clean.csv
python -c "import pandas as pd; df=pd.read_csv('data/master_airquality_clean.csv'); print(f'Rows: {len(df)}, Cols: {len(df.columns)}'); print(df.head())"
```

### Data Quality Notebook (Optional)
```powershell
# Open and run clean.ipynb to verify data quality
jupyter notebook clean.ipynb
```

---

## 2. Model Training & Evaluation

### Single Model Training
```powershell
# Train with Comet ML tracking
python scripts/train_with_comet.py

# Expected output:
# - models/rf_reg.joblib (or similar)
# - models/xgb_reg.json
# - Experiment logged to Comet ML
# - Console output showing metrics (RMSE, MAE, R², MAPE)

# Verify model files
Test-Path models/rf_reg.joblib
Test-Path models/xgb_reg.json
```

### Hyperparameter Tuning
```powershell
# Run hyperparameter search (RandomizedSearchCV/GridSearchCV)
python scripts/hyperparameter_tuning.py

# Expected output: Tuned models with best parameters
# Check console for best parameters and cross-validation scores
```

### Model Evaluation
```powershell
# Evaluate trained model on test set
python scripts/evaluate_model.py

# Expected output:
# - artifacts/evaluation_metrics.json
# - artifacts/test_predictions.csv
# - artifacts/worst_predictions.csv
# Verify artifacts
Test-Path artifacts/evaluation_metrics.json
Get-Content artifacts/evaluation_metrics.json | ConvertFrom-Json
```

### SHAP Interpretability Analysis
```powershell
# Generate SHAP plots
python scripts/shap_analysis.py

# Expected output: SHAP plots (PNG files in artifacts/)
# Verify plots created
Get-ChildItem artifacts/ -Filter *.png
```

### Training Notebook (Alternative)
```powershell
# Open and run full training notebook
jupyter notebook train1.ipynb
```

---

## 3. Model Registry & Versioning

### Register Model
```powershell
# Register trained model with metadata
python scripts/model_registry.py

# Expected: Model registered with version, metadata, and stage (staging/production)
# Check registry files/database
```

---

## 4. Full Pipeline Execution

### Run Complete Pipeline
```powershell
# Execute end-to-end pipeline (ingestion → preprocessing → training → evaluation)
python scripts/pipeline.py

# Expected: All data, models, and artifacts generated sequentially
# Monitor console output for each stage
```

---

## 5. FastAPI Service Testing

### Start API Locally
```powershell
# Start FastAPI server
python main.py

# Expected output:
# - Server running on http://0.0.0.0:8000
# - Uvicorn logs showing startup

# In a new terminal, test endpoints:
```

### Test API Endpoints
```powershell
# Health check
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health

# Metrics endpoint (Prometheus format)
Invoke-RestMethod -Method Get -Uri http://localhost:8000/metrics

# Prediction endpoint (sample payload)
$payload = @{
    "PM10" = 150.0
    "O3" = 45.0
    "CO" = 1.2
    "hour" = 14
    "day_of_week" = 3
    "month" = 6
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body $payload -ContentType "application/json"

# Expected: JSON response with prediction {"prediction": 85.23}
```

---

## 6. Docker & Compose Testing

### Build Docker Image
```powershell
# Build container image
docker build -t pm25-mlops:local .

# Verify image created
docker images | Select-String "pm25-mlops"
```

### Run with Docker Compose
```powershell
# Start all services (FastAPI, Prometheus, Grafana, Alertmanager)
docker-compose up -d

# Check services are running
docker-compose ps

# Expected services:
# - fastapi (port 8000)
# - prometheus (port 9090)
# - grafana (port 3000)
# - alertmanager (port 9093)

# Test FastAPI via Docker
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health

# Access Grafana: http://localhost:3000 (admin/admin)
# Access Prometheus: http://localhost:9090

# Stop services
docker-compose down
```

---

## 7. Monitoring & Drift Detection

### Run Drift Monitor
```powershell
# Execute drift detection (KS test, PSI)
python monitoring/drift_monitor.py

# Expected output:
# - monitoring/reports/drift_report_<timestamp>.json
# - Console output showing drift status per feature

# Verify report
Get-ChildItem monitoring/reports/ -Filter *.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content (Get-ChildItem monitoring/reports/ -Filter *.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName | ConvertFrom-Json
```

### Check Prometheus Metrics
```powershell
# With Docker Compose running, query Prometheus
# Open browser: http://localhost:9090
# Query examples:
# - rate(api_requests_total[5m])
# - histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))
# - up

# Or use API:
Invoke-RestMethod -Method Get -Uri "http://localhost:9090/api/v1/query?query=up"
```

### View Grafana Dashboards
```powershell
# Open Grafana: http://localhost:3000
# Login: admin / admin
# Add Prometheus datasource: http://prometheus:9090
# Import/create dashboards for API performance, drift, model quality
```

---

## 8. Governance & Fairness Testing

### Run Fairness Evaluation
```powershell
# Execute governance notebook (AIF360 fairness metrics)
jupyter notebook governance.ipynb

# Expected: Fairness metrics (DI, SPD) calculated and reported
# Check for disparate impact and statistical parity differences
```

### Verify Documentation
```powershell
# Ensure governance docs are present and up-to-date
Get-Content MODEL_CARD.md
Get-Content DATA_CARD.md
Get-Content GOVERNANCE.md
Get-Content AUDIT_CHECKLIST.md
Get-Content RETRAINING_PLAN.md
Get-Content RISK_ASSESSMENT.md
```

---

## 9. Kubeflow Pipeline (Optional - if Kubernetes available)

### Validate Pipeline Definition
```powershell
# Check pipeline YAML is valid
python validate_pipeline.py

# Expected: No errors, pipeline structure validated
```

### Deploy Kubeflow Pipeline
```powershell
# Deploy pipeline to Kubeflow (requires K8s cluster and Kubeflow installed)
python kubeflow_deploy.py

# Expected: Pipeline uploaded to Kubeflow Pipelines UI
# Monitor runs in Kubeflow dashboard
```

### Run Kubeflow Pipeline
```powershell
# Compile and execute pipeline
python kubeflow_pipeline.py

# Expected: Pipeline executed with all components (data, train, eval, deploy)
# Check Kubeflow UI for run status and artifacts
```

---

## 10. CI/CD Workflows (GitHub Actions)

### Check Workflow Files
```powershell
# List workflow files
Get-ChildItem .github/workflows/ -Filter *.yml

# Expected workflows:
# - ci.yml (linting, testing)
# - mlops.yml (automated training)
# - docker-build.yml (image build/push)
# - deploy-compose.yml (deployment)
```

### Trigger Workflows Locally (Act - optional)
```powershell
# Install act: https://github.com/nektos/act

# Run CI workflow locally
act -j ci

# Note: Requires Docker and act CLI installed
```

---

## 11. End-to-End Integration Test

### Full System Test
```powershell
# 1. Start with clean state
Remove-Item data/master_airquality_clean.csv -ErrorAction SilentlyContinue
Remove-Item models/*.joblib -ErrorAction SilentlyContinue
Remove-Item artifacts/*.json -ErrorAction SilentlyContinue

# 2. Run full pipeline
python scripts/pipeline.py

# 3. Start services
docker-compose up -d

# Wait 10 seconds for services to start
Start-Sleep -Seconds 10

# 4. Test API
$testPayload = @{
    "PM10" = 120.0
    "O3" = 50.0
    "CO" = 1.5
    "hour" = 12
    "day_of_week" = 2
    "month" = 7
} | ConvertTo-Json

$prediction = Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body $testPayload -ContentType "application/json"
Write-Host "Prediction Result: $($prediction | ConvertTo-Json)"

# 5. Check metrics
$metrics = Invoke-RestMethod -Method Get -Uri http://localhost:8000/metrics
Write-Host "Metrics endpoint working: $(if ($metrics) {'✓'} else {'✗'})"

# 6. Run drift detection
python monitoring/drift_monitor.py

# 7. Verify all artifacts
Write-Host "`nVerifying artifacts:"
@(
    "data/master_airquality_clean.csv",
    "models/rf_reg.joblib",
    "artifacts/evaluation_metrics.json",
    "monitoring/reports/drift_report_*.json"
) | ForEach-Object {
    $exists = Test-Path $_
    Write-Host "$_: $(if ($exists) {'✓'} else {'✗'})"
}

# 8. Cleanup
docker-compose down
```

---

## 12. Performance & Load Testing

### API Load Test (Optional)
```powershell
# Install Apache Bench or use PowerShell script
# Simple load test with 100 requests
1..100 | ForEach-Object -Parallel {
    $payload = @{
        "PM10" = 100 + (Get-Random -Maximum 100)
        "O3" = 30 + (Get-Random -Maximum 50)
        "CO" = 1.0 + (Get-Random) 
        "hour" = Get-Random -Maximum 24
        "day_of_week" = Get-Random -Maximum 7
        "month" = Get-Random -Minimum 1 -Maximum 13
    } | ConvertTo-Json
    
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body $payload -ContentType "application/json" | Out-Null
} -ThrottleLimit 10

# Check Prometheus for latency metrics after load test
Invoke-RestMethod -Method Get -Uri "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,%20rate(request_latency_seconds_bucket[1m]))"
```

---

## 13. Troubleshooting & Logs

### Check Logs
```powershell
# Docker Compose logs
docker-compose logs fastapi
docker-compose logs prometheus
docker-compose logs grafana

# Follow logs in real-time
docker-compose logs -f fastapi

# Python script errors
# Check console output when running scripts
# Add --verbose flag if available
```

### Common Issues & Fixes
```powershell
# Issue: Module not found
pip install -r requirements.txt

# Issue: Data file missing
python scripts/data_ingestion.py
python scripts/data_preprocessing.py

# Issue: Model file missing
python scripts/train_with_comet.py

# Issue: Docker port conflict
docker-compose down
# Change ports in docker-compose.yml if needed

# Issue: Prometheus not scraping
# Check prometheus.yml targets
# Verify FastAPI is exposing /metrics

# Issue: Comet/W&B authentication
# Set API keys in environment variables
$env:COMET_API_KEY = "your-key"
```

---

## 14. Optimization Checklist

### Performance Optimization
- [ ] Profile training time: `python -m cProfile -o output.prof scripts/train_with_comet.py`
- [ ] Check model size: `Get-Item models/*.joblib | Select-Object Name, Length`
- [ ] Optimize hyperparameters: Run `scripts/hyperparameter_tuning.py`
- [ ] Cache preprocessed data: Consider saving intermediate results
- [ ] API response time: Check P95 latency in Prometheus

### Code Quality
- [ ] Run linting: `flake8 scripts/ monitoring/ --max-line-length=120`
- [ ] Format code: `black scripts/ monitoring/`
- [ ] Type checking: `mypy scripts/ --ignore-missing-imports`

### Testing Coverage
- [ ] Unit tests: Create `tests/` directory with pytest
- [ ] Integration tests: Test API endpoints
- [ ] Drift tests: Verify drift detection with synthetic data

---

## 15. Final Validation Checklist

```powershell
# Run this comprehensive check
Write-Host "=== MLOps Project Validation ==="

# Data
$dataExists = Test-Path data/master_airquality_clean.csv
Write-Host "✓ Data: $(if ($dataExists) {'PASS'} else {'FAIL'})"

# Models
$modelsExist = (Test-Path models/rf_reg.joblib) -or (Test-Path models/xgb_reg.json)
Write-Host "✓ Models: $(if ($modelsExist) {'PASS'} else {'FAIL'})"

# Artifacts
$artifactsExist = Test-Path artifacts/evaluation_metrics.json
Write-Host "✓ Artifacts: $(if ($artifactsExist) {'PASS'} else {'FAIL'})"

# API (requires service running)
try {
    $health = Invoke-RestMethod -Method Get -Uri http://localhost:8000/health -ErrorAction Stop
    Write-Host "✓ API Health: PASS"
} catch {
    Write-Host "✓ API Health: FAIL (Service not running or unreachable)"
}

# Docker
$dockerRunning = docker ps 2>$null
Write-Host "✓ Docker: $(if ($dockerRunning) {'PASS'} else {'FAIL'})"

# Documentation
$docsExist = (Test-Path MODEL_CARD.md) -and (Test-Path DATA_CARD.md) -and (Test-Path GOVERNANCE.md)
Write-Host "✓ Documentation: $(if ($docsExist) {'PASS'} else {'FAIL'})"

# Monitoring
$driftReports = Get-ChildItem monitoring/reports/ -Filter *.json -ErrorAction SilentlyContinue
Write-Host "✓ Monitoring: $(if ($driftReports) {'PASS'} else {'FAIL'})"

Write-Host "`n=== Validation Complete ==="
```

---

## Quick Start (Recommended Order)

```powershell
# 1. Setup
& "E:\5th SEM Data\AI254TA-Machine Learning Operations(MLOps)\MLOPs_Project\venv\Scripts\Activate.ps1"
pip install -r requirements.txt

# 2. Data Pipeline
python scripts/data_ingestion.py
python scripts/data_preprocessing.py

# 3. Training & Evaluation
python scripts/train_with_comet.py
python scripts/evaluate_model.py

# 4. Full Pipeline Test
python scripts/pipeline.py

# 5. Start Services
docker-compose up -d

# 6. Test API
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health

# 7. Monitor
python monitoring/drift_monitor.py

# 8. Cleanup
docker-compose down
```

---

## Next Steps After Validation

1. **Review Metrics**: Check `artifacts/evaluation_metrics.json` for model performance
2. **Tune Hyperparameters**: Run `scripts/hyperparameter_tuning.py` to improve results
3. **Set Up CI/CD**: Configure GitHub Actions workflows for automation
4. **Deploy to Cloud**: Adapt docker-compose.yml for cloud deployment (Azure/AWS/GCP)
5. **Monitor Production**: Set up Grafana alerts and drift detection schedules
6. **Document Findings**: Update MODEL_CARD.md and GOVERNANCE.md with results

---

For detailed theory and explanations, refer to:
- `docs/01_PROBLEM_STATEMENT.md` through `docs/10_E2E_PIPELINE_DEMO_AND_FINAL_DELIVERABLES.md`
- `README.md` for project overview
- `TUTORIAL_CHECKLIST.md` for implementation status
