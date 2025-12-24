# Project Structure and Organization

## Directory Structure

```
MLOPs_Project/
│
├── .github/
│   └── workflows/              # CI/CD pipelines
│       ├── ci.yml              # Continuous integration
│       ├── mlops.yml           # Model training automation
│       ├── docker-build.yml    # Docker image build
│       └── deploy-compose.yml  # Deployment automation
│
├── artifacts/                  # Evaluation outputs
│   ├── evaluation_metrics.json
│   ├── pred_vs_actual.png
│   ├── residuals.png
│   ├── worst_predictions.csv
│   ├── error_by_hour.png
│   ├── shap_summary.png
│   └── shap_feature_importance.png
│
├── data/                       # Data directory
│   ├── kaggle_csvs/            # Raw data (453 files)
│   ├── raw_combined.csv        # Ingested data
│   ├── master_airquality_clean.csv  # Processed data
│   ├── cities_combined.csv
│   └── stations_combined.csv
│
├── docs/                       # Documentation
│   ├── 01_PROBLEM_STATEMENT.md
│   ├── 02_UNDERSTANDING_MLOPS.md
│   ├── 03_DATA_UNDERSTANDING.md
│   ├── 04_MODEL_BUILDING_EVALUATION.md
│   ├── 05_CI_CD_IMPLEMENTATION.md
│   ├── 06_MONITORING_LIFECYCLE.md
│   ├── APPENDIX_TOOLS.md
│   └── APPENDIX_INSTALLATION.md
│
├── grafana/                    # Grafana configuration
│   ├── dashboards/
│   └── provisioning/
│
├── models/                     # Model artifacts
│   ├── best_pm25_model.pkl    # Production model
│   ├── model_metadata.json    # Model metadata
│   ├── registry/              # Model registry (versioned)
│   │   ├── registry_index.json
│   │   └── pm25_predictor/
│   │       ├── v1/
│   │       ├── v2/
│   │       └── v3/
│   ├── tuning_results.json    # Hyperparameter tuning results
│   └── [chunk_*.joblib files] # Model chunks (if large)
│
├── monitoring/                 # Monitoring scripts
│   ├── drift_monitor.py       # Data drift detection
│   └── reports/               # Drift reports (JSON)
│       └── drift_report_YYYYMMDD_HHMMSS.json
│
├── prometheus/                 # Prometheus configuration
│   ├── alert_rules.yml        # Alerting rules
│   └── alertmanager.yml       # Alertmanager config
│
├── scripts/                    # ML pipeline scripts
│   ├── data_ingestion.py      # Extract data from CSVs
│   ├── data_preprocessing.py  # Clean and transform data
│   ├── train_with_comet.py    # Training with Comet ML
│   ├── evaluate_model.py      # Model evaluation
│   ├── shap_analysis.py       # Interpretability analysis
│   ├── hyperparameter_tuning.py  # Grid/Random search
│   ├── model_registry.py      # Model versioning
│   └── pipeline.py            # Orchestration script
│
├── venv/                       # Virtual environment (not tracked)
│
├── .dockerignore               # Docker ignore file
├── .dvcignore                  # DVC ignore file (placeholder)
├── .gitignore                  # Git ignore file
├── AUDIT_CHECKLIST.md          # Model audit checklist
├── DATA_CARD.md                # Dataset documentation
├── docker-compose.yml          # Multi-service deployment
├── Dockerfile                  # Container definition
├── GOVERNANCE.md               # Governance framework
├── governance.ipynb            # Fairness evaluation notebook
├── governance_report.json      # Fairness metrics
├── IMPLEMENTATION_STATUS.md    # Implementation checklist
├── main.py                     # FastAPI application
├── MODEL_CARD.md               # Model documentation
├── prometheus.yml              # Prometheus config
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
├── RETRAINING_PLAN.md          # Retraining strategy
├── RISK_ASSESSMENT.md          # Risk matrix
├── risk_matrix_5x5.html        # Interactive risk matrix
├── clean.ipynb                 # Data cleaning notebook
└── train1.ipynb                # Training notebook
```

---

## File Descriptions

### Root Directory

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application serving model predictions |
| `Dockerfile` | Container definition for deployment |
| `docker-compose.yml` | Orchestrates FastAPI, Prometheus, Grafana, Alertmanager |
| `requirements.txt` | Python package dependencies |
| `prometheus.yml` | Prometheus scrape configuration |

### Governance Documents

| File | Purpose |
|------|---------|
| `GOVERNANCE.md` | Roles, processes, and governance framework |
| `AUDIT_CHECKLIST.md` | Pre-deployment review checklist |
| `DATA_CARD.md` | Dataset metadata and lineage |
| `MODEL_CARD.md` | Model documentation and use cases |
| `RISK_ASSESSMENT.md` | Risk matrix and mitigation strategies |
| `RETRAINING_PLAN.md` | Criteria and process for retraining |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `data_ingestion.py` | Load and combine raw CSVs |
| `data_preprocessing.py` | Clean, transform, feature engineering |
| `train_with_comet.py` | Train models with Comet ML tracking |
| `evaluate_model.py` | Generate metrics, plots, error analysis |
| `shap_analysis.py` | Model interpretability with SHAP |
| `hyperparameter_tuning.py` | Optimize hyperparameters |
| `model_registry.py` | Version control for models |
| `pipeline.py` | End-to-end orchestration |

### Documentation (`docs/`)

| Document | Purpose |
|----------|---------|
| `01_PROBLEM_STATEMENT.md` | Project objectives and scope |
| `02_UNDERSTANDING_MLOPS.md` | MLOps concepts and lifecycle |
| `03_DATA_UNDERSTANDING.md` | EDA, feature engineering |
| `04_MODEL_BUILDING_EVALUATION.md` | Training and evaluation details |
| `05_CI_CD_IMPLEMENTATION.md` | CI/CD workflows guide |
| `06_MONITORING_LIFECYCLE.md` | Monitoring and lifecycle management |
| `APPENDIX_TOOLS.md` | Tools comparison and pros/cons |
| `APPENDIX_INSTALLATION.md` | Setup and installation guide |

### Monitoring (`monitoring/`)

| File | Purpose |
|------|---------|
| `drift_monitor.py` | Custom KS/PSI drift detection |
| `reports/` | Timestamped drift reports (JSON) |

### CI/CD (`.github/workflows/`)

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Linting, testing on every push |
| `mlops.yml` | Automated training and evaluation |
| `docker-build.yml` | Build and push Docker images |
| `deploy-compose.yml` | Deploy to production |

---

## Data Flow

```
Raw CSVs (data/kaggle_csvs/)
    ↓
data_ingestion.py
    ↓
raw_combined.csv
    ↓
data_preprocessing.py
    ↓
master_airquality_clean.csv
    ↓
train_with_comet.py
    ↓
best_pm25_model.pkl + metadata
    ↓
model_registry.py
    ↓
models/registry/pm25_predictor/vX/
    ↓
main.py (FastAPI)
    ↓
Predictions served via API
```

---

## Artifact Organization

### Model Artifacts
```
models/
├── best_pm25_model.pkl          # Current production model
├── model_metadata.json           # Metadata (features, metrics, date)
├── tuning_results.json           # Hyperparameter tuning results
└── registry/                     # Versioned models
    ├── registry_index.json
    └── pm25_predictor/
        ├── v1/
        │   ├── best_pm25_model.pkl
        │   └── metadata.json
        ├── v2/
        └── v3/ (production)
```

### Evaluation Artifacts
```
artifacts/
├── evaluation_metrics.json       # RMSE, MAE, R², MAPE
├── pred_vs_actual.png            # Scatter plot
├── residuals.png                 # Residual plots
├── error_by_hour.png             # Error analysis by hour
├── worst_predictions.csv         # Top 10 worst predictions
├── shap_summary.png              # SHAP feature importance
├── shap_feature_importance.png   # Bar plot
└── shap_dependence_*.png         # Dependence plots
```

---

## Git Ignore Strategy

### Tracked in Git
- Code (`scripts/`, `monitoring/`, `main.py`)
- Configuration (`requirements.txt`, `docker-compose.yml`)
- Documentation (`docs/`, `*.md`)
- Workflows (`.github/workflows/`)
- Small datasets (`data/cities_combined.csv`, `data/stations_combined.csv`)

### Ignored by Git (`.gitignore`)
- `venv/`, `__pycache__/`
- Large datasets (`data/kaggle_csvs/`, `data/raw_combined.csv`, `data/master_airquality_clean.csv`)
- Model files (`models/*.pkl`, `models/*.joblib`)
- Logs (`*.log`, `wandb/`, `mlruns/`)
- Temporary files (`*.tmp`, `.DS_Store`)
- Environment files (`.env`)

---

## Docker Volumes

```yaml
# docker-compose.yml
volumes:
  - ./prometheus.yml:/etc/prometheus/prometheus.yml
  - ./prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml
  - ./grafana:/etc/grafana/provisioning
  - ./models:/app/models
```

---

## Best Practices Implemented

### 1. Separation of Concerns
- **Data** (`data/`)
- **Code** (`scripts/`, `monitoring/`)
- **Models** (`models/`)
- **Docs** (`docs/`)
- **Config** (root level)

### 2. Versioning
- Code: Git
- Models: Custom registry
- Data: Git (small files), manual versioning (large files)

### 3. Reproducibility
- Pinned dependencies (`requirements.txt`)
- Random seeds in training scripts
- Environment documentation

### 4. Documentation
- Code comments
- Markdown documentation
- Model/Data cards
- Governance documents

---

## Folder Size Estimates

| Directory | Approx. Size |
|-----------|--------------|
| `data/kaggle_csvs/` | ~2 GB |
| `data/master_airquality_clean.csv` | ~1.5 GB |
| `models/` | ~50 MB |
| `venv/` | ~500 MB |
| `artifacts/` | ~10 MB |
| Code + Docs | ~5 MB |

**Total (excluding venv)**: ~3.5 GB

---

## Template for New Projects

This structure can be reused for other ML projects:

```bash
# Copy structure
cp -r MLOPs_Project/ new_project/
cd new_project/

# Clean data and models
rm -rf data/* models/* artifacts/*

# Update docs
vim docs/01_PROBLEM_STATEMENT.md
```

---

## References

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [ML Project Structure Best Practices](https://neptune.ai/blog/ml-project-structure)
