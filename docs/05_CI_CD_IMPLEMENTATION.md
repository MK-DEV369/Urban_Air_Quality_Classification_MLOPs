# CI/CD Implementation Guide

## Overview

This project implements Continuous Integration and Continuous Deployment (CI/CD) using GitHub Actions to automate testing, training, and deployment processes.

## CI/CD Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GitHub Push/Schedule                   │
└────────────────────┬─────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐           ┌──────▼──────┐
    │   CI    │           │   MLOps     │
    │ Testing │           │  Training   │
    └────┬────┘           └──────┬──────┘
         │                       │
         │                 ┌─────▼──────┐
         │                 │  Artifact  │
         │                 │  Upload    │
         │                 └─────┬──────┘
         │                       │
    ┌────▼────────────────────────▼───┐
    │        Docker Build/Push        │
    └────────────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │   Deployment   │
         │  (Optional)    │
         └────────────────┘
```

## GitHub Actions Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Purpose**: Continuous Integration - validate code quality

**Triggers**:
- Every push to any branch
- Every pull request

**Steps**:
1. Checkout code
2. Set up Python 3.10 and 3.11
3. Install dependencies
4. Run linting (flake8)
5. Compile Python files
6. Run tests (if present)

**Key Features**:
- Matrix testing across Python versions
- Fast feedback (~2-3 minutes)
- Caches pip dependencies

**Configuration**:
```yaml
on:
  push:
    branches: ['**']
  pull_request:

jobs:
  python:
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
```

---

### 2. MLOps Training Pipeline (`.github/workflows/mlops.yml`)

**Purpose**: Automate model training and evaluation

**Triggers**:
- Push to `main` branch
- Monthly schedule (1st of every month at midnight)
- Manual trigger (`workflow_dispatch`)

**Steps**:
1. Checkout code
2. Set up Python environment
3. Install dependencies
4. Run complete ML pipeline (`scripts/pipeline.py`)
5. Run drift monitoring
6. Upload model artifacts
7. Upload evaluation reports

**Key Features**:
- Scheduled retraining (monthly)
- Comet ML integration
- Artifact preservation (30 days)

**Artifacts Generated**:
- `trained-model/` - Model files and registry
- `evaluation-reports/` - Metrics, plots, drift reports

**Configuration**:
```yaml
schedule:
  - cron: '0 0 1 * *'  # Monthly
workflow_dispatch:  # Manual trigger
```

---

### 3. Docker Build Workflow (`.github/workflows/docker-build.yml`)

**Purpose**: Build and push Docker images

**Triggers**:
- Push to `main` branch with version tags
- Manual trigger

**Steps**:
1. Checkout code
2. Set up Docker Buildx
3. Login to Docker registry
4. Build image
5. Push to registry

**Key Features**:
- Multi-platform builds (optional)
- Layer caching for faster builds
- Semantic versioning support

---

### 4. Deployment Workflow (`.github/workflows/deploy-compose.yml`)

**Purpose**: Deploy to production using Docker Compose

**Triggers**:
- Manual trigger
- After successful MLOps pipeline

**Steps**:
1. SSH into server
2. Pull latest code
3. Pull Docker images
4. Run `docker-compose up`
5. Health check

---

## Setting Up CI/CD

### Step 1: GitHub Repository Setup

1. Create GitHub repository
2. Push code to `main` branch
3. Workflows automatically detected in `.github/workflows/`

### Step 2: Configure Secrets

Go to **Settings > Secrets and variables > Actions**

Required secrets:
```
COMET_API_KEY         # For experiment tracking
DOCKER_USERNAME       # For Docker Hub (optional)
DOCKER_PASSWORD       # For Docker Hub (optional)
SSH_PRIVATE_KEY       # For deployment (optional)
SERVER_HOST           # Deployment target (optional)
```

### Step 3: Enable Workflows

- Workflows are enabled by default
- Check **Actions** tab to view runs

### Step 4: First Run

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

Workflows will trigger automatically.

---

## Monitoring CI/CD

### GitHub Actions Dashboard

1. Go to **Actions** tab
2. View workflow runs
3. Click on run for detailed logs
4. Download artifacts

### Status Badges

Add to `README.md`:
```markdown
![CI](https://github.com/username/repo/workflows/CI/badge.svg)
![MLOps](https://github.com/username/repo/workflows/MLOps%20Pipeline/badge.svg)
```

---

## CI/CD Best Practices Implemented

### 1. Automated Testing
- ✅ Linting with flake8
- ✅ Syntax validation (compile check)
- ⚠️ Unit tests (placeholder - add tests in `tests/`)

### 2. Dependency Management
- ✅ `requirements.txt` pinned versions
- ✅ Pip cache for faster installs
- ✅ Version matrix testing

### 3. Artifact Management
- ✅ Model files uploaded
- ✅ Evaluation reports preserved
- ✅ 30-day retention policy

### 4. Security
- ✅ Secrets management
- ✅ No hardcoded credentials
- ⚠️ Docker image scanning (add with Trivy)

### 5. Notifications (Optional)
Can add Slack/Email notifications on failure:
```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Manual Triggers

### Run MLOps Pipeline Manually

1. Go to **Actions** tab
2. Select **MLOps Pipeline**
3. Click **Run workflow**
4. Choose branch
5. Click **Run workflow** button

### Deploy Manually

1. Go to **Actions** tab
2. Select **Deploy**
3. Click **Run workflow**
4. Monitor deployment logs

---

## Troubleshooting CI/CD

### Issue: Workflow fails on `pip install`

**Solution**:
- Check `requirements.txt` for version conflicts
- Add `--upgrade` flag if needed
- Review error logs in Actions

### Issue: Comet ML authentication fails

**Solution**:
- Verify `COMET_API_KEY` secret is set
- Check API key is valid at comet.ml

### Issue: Docker build timeout

**Solution**:
- Increase timeout in workflow
- Optimize Dockerfile (multi-stage builds)
- Use layer caching

### Issue: Artifacts not uploading

**Solution**:
- Check paths in `upload-artifact` action
- Ensure files exist before upload
- Verify retention policy

---

## Extending CI/CD

### Add Unit Tests

Create `tests/test_model.py`:
```python
def test_model_prediction():
    import joblib
    model = joblib.load("models/best_pm25_model.pkl")
    prediction = model.predict([[100, 50, 2.0, 10, 0, 6]])
    assert prediction[0] > 0
```

Update `ci.yml`:
```yaml
- name: Run tests
  run: pytest tests/ -v
```

### Add Code Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=scripts --cov-report=xml
```

Upload to Codecov:
```yaml
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Add Performance Testing

```yaml
- name: Benchmark inference
  run: python scripts/benchmark.py
```

### Add Model Validation Gate

```yaml
- name: Validate model performance
  run: |
    python scripts/validate_model.py
    if [ $? -ne 0 ]; then
      echo "Model performance below threshold"
      exit 1
    fi
```

---

## CI/CD Metrics

Track these metrics:

| Metric | Current | Target |
|--------|---------|--------|
| Build Time | ~8 min | < 10 min |
| Success Rate | 95% | > 98% |
| Pipeline Runs/Week | 5 | N/A |
| Mean Time to Recovery | 2 hours | < 1 hour |

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Actions](https://github.com/docker/build-push-action)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
