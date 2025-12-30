
---

## Theory Primer

### MLOps System View
- Treat the ML service as a socio-technical system: data → features → model → service → feedback → governance.
- Separate concerns: data quality, model quality, service reliability, and human processes (approvals/audits).
- Favor reproducibility: fixed seeds, pinned data versions, deterministic preprocessing, and logged metadata.

### Data and Features
- Data generating process matters: drift can stem from sensor changes, policy shifts, or seasonality.
- Guard against leakage: keep temporal and entity splits strict; avoid using post-outcome signals.
- Feature scaling/encoding choices influence stability; track schema and units to avoid silent breaks.

### Evaluation Theory
- Bias–variance–noise: choose model capacity and regularization to balance fit and generalization.
- Metrics: RMSE/MAE for magnitude error, R² for explained variance, MAPE for relative error; segment metrics to catch localized failures.
- Uncertainty: prefer prediction intervals or quantile models when decisions are sensitive to tails.

### Monitoring and Drift
- Data drift: shifts in $P(X)$; detect via KS/PSI and schema checks. Concept drift: shifts in $P(Y|X)$; detect via residual trends.
- SLOs: define latency, availability, and quality thresholds; align alerts to actions (retrain vs rollback).
- Freshness: track model and data age; stale models often degrade before errors spike.

### CI/CD and Deployment
- Immutable artifacts: build once, promote by tag; keep N−1 for rollback.
- Staged promotion: dev → staging → prod; require automated gates (tests, eval) before promotion.
- Canary/blue-green reduce blast radius; observe metrics before full rollout.

### Retraining and Redeployment
- Triggers: drift, performance regression, scheduled cadence, or business events.
- Retraining loop: pull data, train, evaluate, compare to incumbent, document, and promote only with evidence.
- Always validate in staging with the exact image/model intended for prod.

### Governance and Risk
- Fairness: monitor parity across key slices; check both distributional and error parity.
- Documentation: keep Model Card, Data Card, and audit trail updated per promotion.
- Security: protect secrets, validate inputs, and audit dependencies.

### Observability and Alerting
- Metrics: latency/error rate, throughput, drift counts, and quality KPIs.
- Logs: include model/version, request IDs, and minimal input metadata (respect privacy).
- Alerts must be actionable; define runbooks for each class (drift, SLO breach, pipeline failure).

### Reproducibility and Experimentation
- Track code SHA, data hash (DVC), hyperparams, seeds, and environment (Python/package versions).
- Use consistent splits and seeds across experiments; prefer deterministic preprocessing pipelines.

### Safety and Rollback
- Keep rollback playbooks; rehearse rollbacks in staging.
- Avoid irreversible actions during deploy; validate health and a sample prediction before declaring success.
# Installation Guide

## Prerequisites

- **Python**: 3.8 or higher (3.10 recommended)
- **Docker**: Latest version
- **Git**: For version control
- **OS**: Windows, macOS, or Linux

---

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MLOPs_Project
```

---

## Step 2: Set Up Python Environment

### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Option B: Using conda

```bash
conda create -n mlops python=3.10
conda activate mlops
```

---

## Step 3: Install Python Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Note**: If you encounter numpy version conflicts:
```bash
pip install "numpy<2.0" --force-reinstall
```

---

## Step 4: Set Up Comet ML (Optional but Recommended)

1. Create account at [comet.ml](https://www.comet.ml/)
2. Get your API key from Settings
3. Set environment variable:

```bash
# Windows
set COMET_API_KEY=your_api_key_here

# macOS/Linux
export COMET_API_KEY=your_api_key_here
```

Or create a `.comet.config` file:
```ini
[comet]
api_key=your_api_key_here
```

---

## Step 5: Verify Installation

```bash
# Test imports
python -c "import fastapi, sklearn, xgboost, pandas; print('All imports successful!')"

# Check model exists
python -c "import os; print('Model exists:', os.path.exists('models/best_pm25_model.pkl'))"
```

---

## Step 6: Install Docker (for Deployment)

### Windows
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and restart
3. Verify: `docker --version`

### macOS
```bash
brew install --cask docker
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

---

## Step 7: Run the Application

### Option A: Local Development

```bash
# Start API server
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Option B: Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

Access:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## Step 8: Run the ML Pipeline

```bash
# Full pipeline (ingestion -> preprocessing -> training -> evaluation -> SHAP)
python scripts/pipeline.py

# Or run individual steps
python scripts/data_ingestion.py
python scripts/data_preprocessing.py
python scripts/train_with_comet.py
python scripts/evaluate_model.py
python scripts/shap_analysis.py
```

---

## Step 9: Run Drift Monitoring

```bash
python monitoring/drift_monitor.py
```

Results saved to `monitoring/reports/`

---

## Troubleshooting

### Issue: Numpy version conflict

**Solution**:
```bash
pip uninstall numpy
pip install "numpy<2.0"
```

### Issue: Docker permission denied (Linux)

**Solution**:
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### Issue: Port 8000 already in use

**Solution**:
```bash
# Find process
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn main:app --port 8001
```

### Issue: Comet ML API key not found

**Solution**:
Set environment variable or pass directly:
```python
experiment = Experiment(api_key="your_key", project_name="pm25")
```

### Issue: Model file not found

**Solution**:
Download pre-trained model or train from scratch:
```bash
python scripts/train_with_comet.py
```

---

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <repo>
cd MLOPs_Project
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Run pipeline
python scripts/pipeline.py

# 3. Start API
docker-compose up
```

---

## Useful Commands

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Check Docker logs
docker-compose logs -f fastapi

# Restart services
docker-compose restart

# Clean up
docker-compose down --volumes
docker system prune -a

# Run tests (if implemented)
pytest tests/

# Format code
black scripts/ monitoring/
flake8 scripts/ monitoring/
```

---

## Environment Variables

Create a `.env` file:
```env
COMET_API_KEY=your_comet_api_key
MODEL_PATH=models/best_pm25_model.pkl
DATA_PATH=data/master_airquality_clean.csv
LOG_LEVEL=INFO
```

---

## Next Steps

1. Review `docs/01_PROBLEM_STATEMENT.md` for project overview
2. Check `MODEL_CARD.md` for model details
3. Read `GOVERNANCE.md` for governance framework
4. Explore API docs at `/docs` endpoint
5. Set up Grafana dashboards (see `grafana/` folder)

---

## Support

For issues:
1. Check `README.md`
2. Review error logs: `docker-compose logs`
3. Consult documentation in `docs/`
4. Raise an issue on GitHub
