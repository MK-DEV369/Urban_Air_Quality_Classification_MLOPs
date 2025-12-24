# MLOps Tools Comparison

## Tools Used in This Project

### 1. Experiment Tracking: Comet ML

**Purpose**: Track experiments, log metrics, parameters, and visualize results.

| Aspect | Details |
|--------|---------|
| **Pros** | • Free tier available<br>• Excellent UI/UX<br>• Integration with popular frameworks<br>• Code tracking and diffing<br>• Hyperparameter visualization |
| **Cons** | • Requires API key<br>• Cloud-based (privacy concerns for some)<br>• Limited offline capabilities |
| **Alternatives** | MLflow, Weights & Biases, Neptune.ai |

**Setup**:
```bash
pip install comet-ml
export COMET_API_KEY="your_key"
```

**Usage**:
```python
from comet_ml import Experiment
experiment = Experiment(project_name="pm25-airquality")
experiment.log_metric("rmse", 25.5)
```

---

### 2. Model Registry: Custom File-Based

**Purpose**: Version and manage trained model artifacts.

| Aspect | Details |
|--------|---------|
| **Pros** | • No external dependencies<br>• Simple to understand<br>• Full control<br>• Works offline |
| **Cons** | • Manual implementation<br>• No UI<br>• Limited scalability<br>• Basic versioning only |
| **Alternatives** | MLflow Model Registry, AWS SageMaker Model Registry |

**Location**: `scripts/model_registry.py`

---

### 3. CI/CD: GitHub Actions

**Purpose**: Automate testing, training, and deployment.

| Aspect | Details |
|--------|---------|
| **Pros** | • Free for public repos (generous limits for private)<br>• Integrated with GitHub<br>• YAML-based configuration<br>• Large action marketplace |
| **Cons** | • Learning curve for YAML<br>• Limited to GitHub ecosystem<br>• Debugging can be tricky |
| **Alternatives** | GitLab CI, Jenkins, CircleCI, Azure Pipelines |

**Setup**: Workflows in `.github/workflows/`

---

### 4. API Serving: FastAPI

**Purpose**: Serve model predictions via REST API.

| Aspect | Details |
|--------|---------|
| **Pros** | • Fast performance (async support)<br>• Auto-generated docs (Swagger/OpenAPI)<br>• Type hints and validation (Pydantic)<br>• Easy to learn |
| **Cons** | • Python-only<br>• Less mature than Flask<br>• Not as feature-rich as Django |
| **Alternatives** | Flask, Django REST, TensorFlow Serving, Seldon Core |

**Setup**:
```bash
pip install fastapi uvicorn
uvicorn main:app --reload
```

---

### 5. Containerization: Docker & Docker Compose

**Purpose**: Package application with dependencies.

| Aspect | Details |
|--------|---------|
| **Pros** | • Consistent environments<br>• Portable across platforms<br>• Industry standard<br>• Easy multi-service orchestration |
| **Cons** | • Learning curve<br>• Overhead for simple apps<br>• Image size can be large |
| **Alternatives** | Podman, Kubernetes (for orchestration) |

**Setup**:
```bash
docker-compose up --build
```

---

### 6. Monitoring: Prometheus + Grafana

**Purpose**: Collect and visualize operational metrics.

#### Prometheus
| Aspect | Details |
|--------|---------|
| **Pros** | • Time-series database<br>• Powerful query language (PromQL)<br>• Pull-based model<br>• Wide ecosystem |
| **Cons** | • No built-in UI (needs Grafana)<br>• Steep learning curve<br>• Limited long-term storage |
| **Alternatives** | InfluxDB, Datadog, New Relic |

#### Grafana
| Aspect | Details |
|--------|---------|
| **Pros** | • Beautiful dashboards<br>• Many data source integrations<br>• Alerting support<br>• Open source |
| **Cons** | • Can be resource-intensive<br>• Configuration complexity |
| **Alternatives** | Kibana, Tableau |

---

### 7. Alerting: Prometheus Alertmanager

**Purpose**: Send alerts on threshold violations.

| Aspect | Details |
|--------|---------|
| **Pros** | • Integrates with Prometheus<br>• Flexible routing<br>• Supports multiple channels (email, Slack, PagerDuty) |
| **Cons** | • Configuration can be verbose<br>• Requires separate deployment |
| **Alternatives** | PagerDuty, Opsgenie, custom webhooks |

**Location**: `prometheus/alert_rules.yml`

---

### 8. Drift Detection: Custom KS/PSI

**Purpose**: Detect data distribution changes.

| Aspect | Details |
|--------|---------|
| **Pros** | • No external dependencies<br>• Fast computation<br>• Statistical rigor (KS test, PSI) |
| **Cons** | • Manual implementation<br>• No UI<br>• Limited advanced features |
| **Alternatives** | Evidently AI, NannyML, Alibi Detect |

**Location**: `monitoring/drift_monitor.py`

---

### 9. Fairness Evaluation: AIF360

**Purpose**: Assess model bias and fairness.

| Aspect | Details |
|--------|---------|
| **Pros** | • Comprehensive fairness metrics<br>• Open source (IBM)<br>• Mitigation algorithms included |
| **Cons** | • Complex API<br>• Heavy dependencies<br>• Slow for large datasets |
| **Alternatives** | Fairlearn (Microsoft), What-If Tool (Google) |

**Setup**:
```bash
pip install aif360
```

**Location**: `governance.ipynb`

---

### 10. Interpretability: SHAP

**Purpose**: Explain model predictions.

| Aspect | Details |
|--------|---------|
| **Pros** | • Theory-backed (Shapley values)<br>• Works with any model<br>• Great visualizations<br>• Local and global explanations |
| **Cons** | • Slow for large datasets<br>• Can be memory-intensive<br>• Requires numpy<2.0 (compatibility issues) |
| **Alternatives** | LIME, InterpretML, ELI5 |

**Setup**:
```bash
pip install shap matplotlib
```

**Location**: `scripts/shap_analysis.py`

---

## Tools We Explicitly Avoided

### MLflow
**Why avoided**: User requested to not use MLflow. Replaced with Comet ML for tracking and custom registry for model versioning.

### DVC (Data Version Control)
**Why avoided**: User requested to not use DVC. Using Git for code and metadata versioning instead.

### Evidently AI
**Why avoided**: User requested to not use Evidently. Implemented custom drift detection using statistical tests.

---

## Potential Additions (Not Implemented)

### Kubeflow
- **Purpose**: Pipeline orchestration on Kubernetes
- **Status**: Mentioned by user but not implemented
- **Use Case**: For complex, distributed training pipelines

### Feature Store (e.g., Feast)
- **Purpose**: Centralized feature management
- **Status**: Not needed for this project size
- **Use Case**: When features are shared across many models

### Model Serving (Seldon Core, KServe)
- **Purpose**: Advanced serving with A/B testing, canary deployments
- **Status**: FastAPI sufficient for current needs
- **Use Case**: High-scale production deployments

---

## Tool Selection Decision Matrix

| Requirement | Tool Chosen | Reasoning |
|-------------|-------------|-----------|
| Experiment Tracking | Comet ML | User requirement (no MLflow) + good UX |
| Model Versioning | Custom Registry | Lightweight, no external deps |
| Data Versioning | Git (no DVC) | User requirement |
| CI/CD | GitHub Actions | Integrated with repo |
| API Serving | FastAPI | Fast, modern, auto-docs |
| Containerization | Docker | Industry standard |
| Metrics Collection | Prometheus | Time-series, pull-based |
| Visualization | Grafana | Best-in-class dashboards |
| Drift Detection | Custom KS/PSI | User requirement (no Evidently) |
| Fairness | AIF360 | Comprehensive metrics |
| Interpretability | SHAP | Theory-backed, popular |

---

## Installation Summary

```bash
# Core ML
pip install pandas numpy scikit-learn xgboost joblib

# Experiment Tracking
pip install comet-ml

# API Serving
pip install fastapi uvicorn

# Monitoring
pip install prometheus-client

# Interpretability & Fairness
pip install shap matplotlib seaborn aif360

# Statistical Tests
pip install scipy

# Constraint: numpy<2.0 (compatibility)
pip install "numpy<2.0"
```

See `requirements.txt` for complete list.
