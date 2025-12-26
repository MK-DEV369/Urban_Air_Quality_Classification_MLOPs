# MLOps Urban Air Quality Prediction Dashboard

An end-to-end MLOps project for predicting urban air quality (PM2.5 levels) using machine learning models, deployed with FastAPI, monitored via Prometheus, and tracked with Weights & Biases.

## 🚀 Features

- **Data Pipeline**: Automated data cleaning and preprocessing combining multiple air quality datasets
- **Machine Learning Models**: Comparative training of Linear Regression, Random Forest, and XGBoost models
- **Kubeflow Pipeline Orchestration**: Complete ML workflow orchestration using Kubeflow Pipelines
- **FastAPI Deployment**: RESTful API for real-time PM2.5 predictions
- **Monitoring & Observability**: Prometheus metrics + Grafana dashboard for API performance tracking
- **Experiment Tracking**: Weights & Biases integration for model versioning and logging
- **Model Governance**: Fairness and bias analysis using AIF360
- **Containerized Deployment**: Docker-based setup for easy deployment
- **CI/CD**: GitHub Actions workflows for CI + Docker build/push and optional deploy

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Weights & Biases account (for experiment tracking)
- Git

## 🛠️ System Requirements

### Core Dependencies
- fastapi
- uvicorn
- numpy
- pandas
- joblib
- scikit-learn
- xgboost
- prometheus-client

### Additional Tools
- Docker
- Weights & Biases CLI
- Jupyter Notebook (for data processing)
- Kubeflow Pipelines SDK (for pipeline orchestration)

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MLOPs_Project
```

### 2. Data Preparation

#### Combine Cities and Industries Datasets
The project requires combining air quality data from cities and industrial sources:

1. Ensure you have the following datasets in the `data/` folder:
   - `cities_combined.csv` (cities air quality data)
   - `stations_csvs/` directory with individual station CSV files

2. Run the data cleaning notebook:
   ```bash
   jupyter notebook clean.ipynb
   ```
   Or execute the cells in sequence to:
   - Load and merge station data into `stations_combined.csv`
   - Combine cities and stations data
   - Clean missing values and create time features
   - Save the final cleaned dataset as `data/master_airquality_clean.csv`

### 3. Model Training

1. Open and run the training notebook:
   ```bash
   jupyter notebook train1.ipynb
   ```

2. The notebook will:
   - Load the cleaned dataset
   - Train Linear Regression, Random Forest, and XGBoost models
   - Log results to Weights & Biases
   - Save the best performing model as `models/best_pm25_model.pkl`

3. Combine Random Forest model chunks:
   After training, run the combine script to merge the trained Random Forest chunks into a single ensemble model:
   ```bash
   python models/combine_joblib.py
   ```
   This creates `models/rf_reg.joblib` for further steps.

**Note**: Ensure you have a Weights & Biases account and run `wandb login` before training.

### 4. Kubeflow Pipeline Orchestration (Optional)

For production-scale ML workflow orchestration:

1. Install Kubeflow Pipelines SDK:
   ```bash
   pip install kfp==2.7.0
   ```

2. Compile the pipeline:
   ```bash
   python kubeflow_pipeline.py
   ```
   This generates `pm25_pipeline.yaml` with 5 components:
   - Data Ingestion
   - Data Preprocessing
   - Model Training (XGBoost)
   - Model Evaluation
   - Drift Detection

3. Deploy to Kubeflow (requires Kubeflow installation):
   ```bash
   python kubeflow_deploy.py --host http://localhost:8080
   ```

See [KUBEFLOW_QUICKSTART.md](KUBEFLOW_QUICKSTART.md) for detailed instructions.

### 5. Docker Deployment

1. Build and start the services:
   ```bash
   docker-compose up --build
   ```

2. The following services will be available:
   - **FastAPI Application**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Prometheus Monitoring**: http://localhost:9090
   - **Grafana Dashboards**: http://localhost:3000 (datasource + dashboard auto-provisioned)

## 🔧 Usage

### API Endpoints

#### Health Check
```bash
GET /
```
Response: `{"message": "PM2.5 Prediction API is running!"}`

#### Make Predictions
```bash
POST /predict
Content-Type: application/json

{
  "PM10": 45.2,
  "O3": 25.1,
  "CO": 0.8,
  "hour": 14,
  "dayofweek": 2,
  "month": 7
}
```
Response:
```json
{
  "PM25_prediction": 32.45,
  "model_used": "XGBRegressor"
}
```

#### Prometheus Metrics
```bash
GET /metrics
```
Returns Prometheus-formatted metrics for monitoring.

#### Grafana Dashboard
- Open Grafana: http://localhost:3000
- Default login (Grafana defaults): `admin` / `admin` (you’ll be prompted to change it)
- Dashboard: **FastAPI MLOps (PM2.5) - Overview**

### Monitoring Dashboard

- **Prometheus**: Access at http://localhost:9090 to view API metrics
- **Weights & Biases**: View experiment runs and model artifacts in your W&B dashboard

## 📊 Project Outcomes

### Trained Models
- Best performing model saved as `models/best_pm25_model.pkl`
- Model comparison results logged to Weights & Biases
- Performance metrics (RMSE, R²) available for all trained models

### API Performance
- Real-time PM2.5 predictions via REST API
- Automatic metrics collection for request count, latency, and prediction volume
- Interactive API documentation at `/docs`

### Data Insights
- Cleaned and merged air quality dataset (`data/master_airquality_clean.csv`)
- Time-based features (hour, day of week, month) for temporal analysis
- Governance report with fairness analysis (`governance_report.json`)

### Monitoring & Observability
- Prometheus metrics for API health and performance (`/metrics`)
- Grafana dashboard provisioned from `grafana/`
- Experiment tracking with model versioning
- Bias and fairness assessment using AIF360

### Model Documentation
- Model Card template: `MODEL_CARD.md`

### Drift Detection (Optional)
- Evidently drift report scaffold: `monitoring/evidently_drift_report.py`
- Optional scheduled workflow: `.github/workflows/drift-report.yml` (uploads HTML report artifact)

## 🏗️ Project Structure

```
MLOPs_Project/
├── data/
│   ├── cities_combined.csv
│   ├── stations_combined.csv
│   ├── master_airquality_clean.csv
│   └── stations_csvs/
├── models/
│   ├── best_pm25_model.pkl
│   ├── linear_reg.joblib
│   ├── rf_reg.joblib
│   └── xgb_reg.json
├── clean.ipynb          # Data cleaning and preprocessing
├── train1.ipynb         # Model training and W&B logging
├── governance.ipynb     # Model governance and fairness analysis
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Multi-service deployment
├── Dockerfile           # Application containerization
├── prometheus.yml       # Monitoring configuration
└── README.md
```

## What features are still NOT utilized (future work)

These are common MLOps features not fully implemented in this repo yet:

- **Formal data versioning (DVC)**: data is stored in `data/` without DVC tracking/remotes
- **Model registry (formal)**: models are stored locally / W&B artifacts (no dedicated registry like MLflow Registry)
- **Automated retraining pipeline**: training is notebook-driven (no scheduled retrain + promotion)
- **ML performance monitoring in production**: operational metrics exist, but no continuous ground-truth monitoring

## Viva/Exam-ready tool status (updated)

| MLOps Layer         | Tools Used                         | Status     |
| ------------------- | ---------------------------------- | ---------- |
| Data Processing     | Pandas, NumPy                      | ✅ Complete |
| Feature Engineering | Manual (time features)             | ✅ Complete |
| Model Training      | Scikit-learn, XGBoost              | ✅ Complete |
| Experiment Tracking | Weights & Biases                   | ✅ Complete |
| Deployment          | FastAPI, Docker, Docker Compose    | ✅ Complete |
| Monitoring          | Prometheus + Grafana               | ✅ Complete |
| Governance          | AIF360                              | ⚠ Partial  |
| CI/CD               | GitHub Actions                     | ✅ Complete |
| Drift Detection     | EvidentlyAI (scaffold + workflow)  | ⚠ Partial  |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or support, please open an issue in the repository.

## CI/CD with GitHub Actions

This repo includes ready‑to‑use GitHub Actions for CI, Docker image build/push, and optional deploy via Docker Compose over SSH.

### Workflows
- CI: runs on every push/PR. See [.github/workflows/ci.yml](.github/workflows/ci.yml)
   - Sets up Python 3.10/3.11, installs `requirements.txt`
   - Lints with `flake8`, compiles all `.py` files
   - Runs `pytest` only if a `tests/` folder exists

- Train/Evaluate (manual/scheduled): runs evaluation + interpretability and uploads artifacts. See [.github/workflows/train-evaluate.yml](.github/workflows/train-evaluate.yml)
   - If `data/master_airquality_clean.csv` is not in the repo, provide a `data_url` input when manually triggering
   - Uploads `artifacts/` (metrics JSON + test predictions + feature importance)

- Docker Build and Push (GHCR): runs on `main` and on demand. See [.github/workflows/docker-build.yml](.github/workflows/docker-build.yml)
   - Builds the image from `Dockerfile`
   - Pushes to GHCR: `ghcr.io/<owner>/mlops-project` with `latest`, branch, tag, and `sha` tags

- Deploy via SSH (optional): runs after a successful image push or on demand. See [.github/workflows/deploy-compose.yml](.github/workflows/deploy-compose.yml)
   - SSHes into your host, logs in to GHCR, and runs `docker compose pull` + `up -d`

### Required repository settings/secrets
Add these in GitHub → Settings → Secrets and variables → Actions → New repository secret:

- GHCR_PAT: Personal Access Token with `read:packages` (used on the remote host to `docker login ghcr.io`)
- SSH_HOST: Public hostname or IP of your server
- SSH_USER: SSH username
- SSH_PRIVATE_KEY: Private key for SSH (PEM text)
- SSH_PORT: Optional; default `22`
- DEPLOY_DIR: Absolute directory on the server where your `docker-compose.yml` lives

No extra secret is required for pushing to GHCR from Actions; the workflow uses `${{ secrets.GITHUB_TOKEN }}` with `packages: write` permission.

### Using the GHCR image in docker-compose
Update your service to use the pushed image instead of building locally (example):

```yaml
services:
   app:
      image: ghcr.io/<owner>/mlops-project:latest
      # remove "build:" if present and keep your env/ports/volumes as is
      ports:
         - "8000:8000"
      env_file:
         - .env
```

### Manual runs
- CI: Actions → CI → Run workflow
- Build/Push: Actions → Docker Build and Push (GHCR) → Run workflow
- Deploy: Actions → Deploy (Docker Compose via SSH) → Run workflow (optionally specify `image_tag`)

### Notes
- The `.dockerignore` excludes large/local artifacts (e.g., `data/`, `wandb/`, notebooks). If your container needs local files, remove them from `.dockerignore`.
- If your app loads models from `models/`, they are included by default.
- If you add tests later, place them under `tests/` and CI will run them automatically.

## Report Deliverables (Rubric Support)

- Risk matrix explanation: [docs/risk_matrix.md](docs/risk_matrix.md)
- Dataset references & licensing: [docs/dataset_references.md](docs/dataset_references.md)
- Data Card: [DATA_CARD.md](DATA_CARD.md)
- Retraining plan: [docs/retraining_plan.md](docs/retraining_plan.md)
- Audit checklist: [AUDIT_CHECKLIST.md](AUDIT_CHECKLIST.md)
- DVC setup guide: [docs/dvc_setup.md](docs/dvc_setup.md)
- Model registry approach: [docs/model_registry.md](docs/model_registry.md)

### Local evaluation / interpretability

```bash
python scripts/evaluate_model.py --data data/master_airquality_clean.csv --model models/best_pm25_model.pkl --outdir artifacts
python scripts/interpretability.py --model models/best_pm25_model.pkl --outdir artifacts
```

### Alerting
- Prometheus rules: `prometheus/alert_rules.yml`
- Alertmanager: exposed at http://localhost:9093 when using docker compose
