
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
# MLOps Urban Air Quality Prediction Dashboard

An end-to-end MLOps project for predicting urban air quality (PM2.5 levels) using machine learning models, deployed with FastAPI, monitored via Prometheus, and tracked with Weights & Biases.

## Problem Statement

Urban air pollution, particularly PM2.5 (particulate matter ≤2.5µm), poses serious health risks. Timely prediction of PM2.5 helps public health officials issue warnings, citizens plan outdoor activity, and city planners enact mitigation. The challenge is to build a robust, production-ready ML system that predicts PM2.5 using co-pollutants (e.g., PM10, O₃, CO) and temporal features, and to operate it reliably with MLOps practices.

See detailed context in [docs/01_PROBLEM_STATEMENT.md](docs/01_PROBLEM_STATEMENT.md).

## Objectives

### Primary Objectives
- **Accurate predictions**: Achieve test RMSE < 30 µg/m³, MAE < 20 µg/m³, R² > 0.75
- **Operational API**: Serve real-time predictions with P95 latency < 100ms
- **MLOps foundations**: Containerized deployment, CI/CD, experiment tracking, monitoring

### Secondary Objectives
- **Fairness**: Evaluate bias across temporal segments (weekday/weekend) with AIF360
- **Interpretability**: Provide SHAP-based explanations and feature importance
- **Drift handling**: Monitor data/model drift and automate retraining triggers
- **Reliability**: Target API uptime > 99.5% with alerting (Prometheus/Grafana)

### Scope
- In scope: PM2.5 point predictions from pollutant and time features; batch and real-time inference; orchestration via Kubeflow; monitoring; governance docs
- Out of scope: Multistep forecasting, causal inference, multi-city production rollout, mobile apps

### Success Criteria
- Model meets performance targets on a holdout test set
- API meets latency and uptime targets with monitoring in place
- CI/CD automates build/test/deploy for reproducibility
- Governance documentation completed (Model Card, Audit, Report) — see [GOVERNANCE.md](GOVERNANCE.md)

## Project Scope

### In Scope
- Predict PM2.5 levels using input features: `PM10`, `O3`, `CO`, and temporal features (`hour`, `dayofweek`, `month`).
- Support both batch and real-time inference via FastAPI.
- Automate workflows with Kubeflow Pipelines for ingestion, preprocessing, training, evaluation, and drift checks.
- Continuous monitoring of API performance and drift using Prometheus/Grafana plus scheduled reports.
- Version models and track experiments; maintain governance documentation and audit artifacts.

### Out of Scope
- Multistep forecasting of future PM2.5 trajectories.
- Causal inference explaining pollution sources.
- Broad multi‑city production rollout beyond the provided datasets.
- Mobile application development.

## Evaluation Metrics

### Model Performance
- **RMSE** (< 30 µg/m³): Penalizes large errors; primary accuracy metric.
- **MAE** (< 20 µg/m³): Average absolute error; easy to interpret in µg/m³.
- **R²** (> 0.75): Variance explained; indicates model fit quality.
- **MAPE** (< 40%): Percentage error; business‑friendly comparison across ranges.

Formulas:
- RMSE: $\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$
- MAE: $\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i|$
- R²: $R^2 = 1 - \frac{\sum(\hat{y}-y)^2}{\sum(y-\bar{y})^2}$
- MAPE: $\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{\hat{y}_i - y_i}{y_i}\right|$

### Operational Metrics
- **API Latency (P95)** < 100ms (tracked in Prometheus/Grafana).
- **API Uptime** > 99.5% with alerting rules in [prometheus/alert_rules.yml](prometheus/alert_rules.yml).
- **Error Rate** < 1% across endpoints; investigate spikes via logs/alerts.
- **Drift Detection**: Daily checks with KS/PSI statistics; trigger retraining plan in [RETRAINING_PLAN.md](RETRAINING_PLAN.md).

### Fairness & Governance
- **Disparate Impact** target: 0.8–1.25 across segments (weekday/weekend) using AIF360.
- **Statistical Parity Difference** target: < 0.1.
- **Documentation & Audit**: Maintain [MODEL_CARD.md](MODEL_CARD.md), complete [AUDIT_CHECKLIST.md](AUDIT_CHECKLIST.md), archive [governance_report.json](artifacts/evaluation_metrics.json) outputs; follow roles/processes in [GOVERNANCE.md](GOVERNANCE.md).

## Tools & Environment Setup

### Toolchain Overview
- **Language & Frameworks**: Python 3.8+, FastAPI, NumPy/Pandas, scikit‑learn, XGBoost, Joblib
- **Orchestration**: Kubeflow Pipelines (compile/deploy via `kubeflow_pipeline.py`, `kubeflow_deploy.py`)
- **Monitoring**: Prometheus (metrics), Grafana (dashboards) — configured via [prometheus.yml](prometheus.yml) and [grafana/](grafana)
- **Experiment Tracking**: Weights & Biases (W&B)
- **Notebooks**: Jupyter for data cleaning/training ([clean.ipynb](clean.ipynb), [train1.ipynb](train1.ipynb))
- **CI/CD**: GitHub Actions for build/test/container image
- **Containerization**: Docker, Docker Compose ([docker-compose.yml](docker-compose.yml))

### Local Python Environment (Windows)
Use PowerShell and a virtual environment for isolation.

```powershell
python -m venv venv
.\n+venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Optional extras (if not already in `requirements.txt`):

```powershell
pip install kfp==2.7.0  # Kubeflow Pipelines SDK
pip install shap        # Interpretability
pip install aif360      # Fairness (requires dependencies)
```

Authenticate W&B (for experiment tracking):

```powershell
wandb login
```

Tip: If PowerShell blocks script execution, run as Administrator:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Services via Docker Compose
Build and start application + monitoring stack:

```powershell
docker-compose up --build -d
```

Exposed endpoints:
- FastAPI: http://localhost:8000 (OpenAPI at `/docs`)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

To stop services:

```powershell
docker-compose down
```

### Kubeflow Pipeline (Optional)
Compile the pipeline and generate `pm25_pipeline.yaml`:

```powershell
pip install kfp==2.7.0
python kubeflow_pipeline.py
```

Deploy to a running Kubeflow instance:

```powershell
python kubeflow_deploy.py --host http://localhost:8080
```

### Notebooks and Artifacts
- Data cleaning: run [clean.ipynb](clean.ipynb) → outputs [data/master_airquality_clean.csv](data/master_airquality_clean.csv)
- Training: run [train1.ipynb](train1.ipynb) → saves models under [models/](models)
- Evaluation/interpretability: scripts in [scripts/](scripts) emit reports under [artifacts/](artifacts)

### Quick Sanity Checks
Health check:

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:8000/
```

Prediction example:

```powershell
$body = @{PM10=45.2; O3=25.1; CO=0.8; hour=14; dayofweek=2; month=7} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body $body -ContentType 'application/json'
```

## Data Collection & Exploration Techniques

### Data Sources and Volume
- **Origin**: Kaggle India Air Quality dataset (2015–2020) plus station‑level CSVs under [data/kaggle_csvs/](data/kaggle_csvs).
- **Coverage**: ~453 monitoring stations across multiple states (e.g., Delhi 29, Bihar 35, Chhattisgarh 14, Andhra Pradesh 10).
- **Scale**: ~14M raw records; ~12M after cleaning.

### Ingestion & Consolidation
- **Ingestion**: Load per‑station CSVs and city aggregates via [scripts/data_ingestion.py](scripts/data_ingestion.py).
- **Consolidation**: Merge station and city datasets; deduplicate; standardize timestamps.
- **Outputs**: Cleaned master dataset at [data/master_airquality_clean.csv](data/master_airquality_clean.csv).

### Data Quality Profiling
- **Missingness**: Many pollutants have 30–60% missingness; target `PM2.5` ~15% missing.
- **Units/duplication**: Mixed units, duplicate columns across sources; resolve by type conversion and selective retention.
- **Outliers**: Sensor spikes up to 999 µg/m³; remove extremes via percentile filtering (1st–99th).

### Preprocessing Pipeline
Implemented in [scripts/data_preprocessing.py](scripts/data_preprocessing.py) and [clean.ipynb](clean.ipynb):
- Parse `Timestamp` and drop invalid rows.
- Type‑cast numeric features: `PM2.5`, `PM10`, `O3`, `CO`.
- Impute feature missing values with median; drop rows with missing target.
- Remove extreme `PM2.5` outliers using quantiles.

### Feature Engineering
- **Pollutants**: `PM10` (strong positive correlation), `CO` (industrial signal), `O3` (weak negative correlation).
- **Temporal**: `hour`, `dayofweek`, `month` to capture diurnal/weekly/seasonal patterns.
- **Rationale**: Availability, predictive strength, interpretability, and real‑time feasibility.

### Exploratory Analysis (EDA)
- **Temporal patterns**: Rush‑hour peaks (7–9 AM, 6–8 PM); winter highs; monsoon lows.
- **Spatial patterns**: Delhi highest averages; coastal cities lower; industrial zones elevated `CO`/`PM10`.
- **Correlations**: `PM10` ≈ 0.87, `CO` ≈ 0.42, `NO2` ≈ 0.38, `O3` ≈ −0.15 with `PM2.5`.
- **Artifacts**: See visualizations and CSVs under [artifacts/](artifacts).

### Versioning & Governance
- **Storage**: Raw station files in [data/kaggle_csvs/](data/kaggle_csvs); processed output in [data/master_airquality_clean.csv](data/master_airquality_clean.csv).
- **Documentation**: Dataset description and quality checks in [DATA_CARD.md](DATA_CARD.md); governance and fairness in [GOVERNANCE.md](GOVERNANCE.md).
- **Orchestration**: End‑to‑end via [scripts/pipeline.py](scripts/pipeline.py) (ingestion → preprocessing → output).

See detailed guide in [docs/03_DATA_UNDERSTANDING.md](docs/03_DATA_UNDERSTANDING.md).

## Data Profiling, Cleaning & Preprocessing

### Profiling Highlights
- **Target `PM2.5`**: Mean ≈ 89.3 µg/m³, Median ≈ 60.1 µg/m³, Std ≈ 82.5 µg/m³, Range 0–999 µg/m³ (contains outliers), Missing ≈ 15%.
- **Correlations (with `PM2.5`)**: `PM10` ≈ 0.87 (strong), `CO` ≈ 0.42 (moderate), `NO2` ≈ 0.38 (moderate), `O3` ≈ −0.15 (weak negative).
- **Missingness**: Many raw features have 30–60% missing; selected feature set minimizes missingness post-imputation.
- **Artifacts**: Visuals and CSV summaries under [artifacts/](artifacts) and notebook outputs in [clean.ipynb](clean.ipynb).

### Cleaning Steps (Implemented)
1. **Timestamp parsing**: Normalize `Timestamp`; drop invalid rows.
2. **Type conversion**: Cast numeric features (`PM2.5`, `PM10`, `O3`, `CO`) to numeric with coercion.
3. **Feature engineering (temporal)**: Add `hour`, `dayofweek`, `month` from parsed timestamps.
4. **Missing value handling**: Drop rows with missing target; median-impute selected features.
5. **Outlier removal**: Filter extreme `PM2.5` values using 1st–99th percentile bounds.

References: [scripts/data_preprocessing.py](scripts/data_preprocessing.py), [clean.ipynb](clean.ipynb).

### Final Feature Set (for Modeling)
- **Pollutants**: `PM10`, `CO`, `O3`
- **Temporal**: `hour`, `dayofweek`, `month`

Rationale: Strong predictive signal, lower missingness post-cleaning, interpretability, and real-time feasibility.

### End-to-End Data Pipeline
- **Ingestion**: [scripts/data_ingestion.py](scripts/data_ingestion.py)
- **Preprocessing**: [scripts/data_preprocessing.py](scripts/data_preprocessing.py)
- **Output**: [data/master_airquality_clean.csv](data/master_airquality_clean.csv)

Run steps (Windows PowerShell):

```powershell
python scripts/data_ingestion.py
python scripts/data_preprocessing.py
```

### Quality Checks & Validation
- Validate timestamp parsing and non-null `PM2.5` in the final dataset.
- Confirm numeric dtypes and acceptable ranges for `PM10`, `CO`, `O3`.
- Review distribution plots and correlation heatmaps (EDA) before training.
- Spot-check drift and missingness trends in scheduled reports.

See the detailed methodology and statistics in [docs/03_DATA_UNDERSTANDING.md](docs/03_DATA_UNDERSTANDING.md).

## Feature Extraction & Selection

### Engineered Features (Extraction)
- **Temporal features**: Derived from `Timestamp` — `hour`, `dayofweek` (0=Mon), `month` — to capture diurnal, weekly, and seasonal patterns.
- **Pollutant features**: Normalize and cast numeric values for `PM10`, `CO`, `O3` to ensure consistent numeric dtypes.
- **Outlier handling**: Remove extreme `PM2.5` targets via percentile bounds; retain plausible pollutant ranges.
- **Missingness controls**: Drop rows with missing `PM2.5`; median-impute selected input features to stabilize training and inference.

Implementation references: [scripts/data_preprocessing.py](scripts/data_preprocessing.py), [clean.ipynb](clean.ipynb).

### Selection Strategy
- **Availability & coverage**: Prefer features consistently present across stations/cities; deprioritize high‑missingness fields.
- **Predictive signal**: Use correlation screening (e.g., `PM10` ≈ 0.87, `CO` ≈ 0.42, `O3` ≈ −0.15 with `PM2.5`) to shortlist.
- **Interpretability & operations**: Favor features explainable to stakeholders and feasible to collect in near real‑time.
- **Model‑aware checks**: Validate with embedded importance (XGBoost gain/weight), plus post‑hoc SHAP analysis.
- **Robustness**: Assess stability across temporal segments (weekday vs weekend) and seasons.

Selected features: `PM10`, `CO`, `O3`, `hour`, `dayofweek`, `month`.

Excluded features: Meteorological variables (Temp, RH, WS, etc.) and VOCs (Benzene/Toluene/Xylene) due to high missingness/limited coverage; additional pollutants (NO2, SO2) often redundant or sparsely available.

### Validation & Interpretability
- **SHAP analysis**: Summaries and dependence plots in [artifacts/](artifacts) (e.g., `shap_summary.png`).
- **Post‑training importance**: `PM10` dominates; `CO` and `O3` contribute secondary signals; temporal features capture periodicity.
- **Sanity checks**: Residual and error‑by‑hour plots to confirm temporal usefulness.

Run interpretability (example):

```powershell
python scripts/shap_analysis.py --model models/best_pm25_model.pkl --data data/master_airquality_clean.csv --outdir artifacts
```

### Operational Considerations
- **Inference readiness**: Temporal features computed from request timestamp; pollutants provided by upstream sensors.
- **Scaling**: Tree‑based models do not require feature scaling; linear baselines may benefit from standardization.
- **Drift sensitivity**: Monitor distribution shifts for `PM10/CO/O3` and temporal segments; retrain when thresholds are exceeded.

## Model Building

### Workflow Overview
- **Input**: Cleaned dataset [data/master_airquality_clean.csv](data/master_airquality_clean.csv) with features `PM10`, `CO`, `O3`, `hour`, `dayofweek`, `month` and target `PM2.5`.
- **Split**: Temporal Train/Val/Test ≈ 60/20/20 without shuffling to prevent leakage.
- **Train**: Fit Linear Regression, Random Forest, and XGBoost.
- **Evaluate**: Compute RMSE/MAE/R²/MAPE; generate plots (pred‑vs‑actual, residuals, error by hour).
- **Persist**: Save best model and metadata under [models/](models) and metrics under [artifacts/](artifacts).

### Running Training
- Notebook flow:

```powershell
jupyter notebook train1.ipynb
```

- Scripted flow with experiment tracking:

```powershell
python scripts\train_with_comet.py --data data\master_airquality_clean.csv --outdir artifacts
```

After training, combine Random Forest chunks if applicable:

```powershell
python models\combine_joblib.py
```

### Model Configurations (Examples)
- **Linear Regression**: `LinearRegression()` (no hyperparameters).
- **Random Forest**: `n_estimators=100`, `max_depth=20`, `min_samples_split=2`, `random_state=42`.
- **XGBoost**: `n_estimators=300`, `learning_rate=0.05`, `max_depth=7`, `subsample=0.9`, `colsample_bytree=0.9`, `tree_method='hist'`, `objective='reg:squarederror'`.

### Evaluation & Artifacts
- Metrics JSON: [artifacts/evaluation_metrics.json](artifacts/evaluation_metrics.json)
- Predictions CSVs: [artifacts/test_predictions.csv](artifacts/test_predictions.csv), [artifacts/worst_predictions.csv](artifacts/worst_predictions.csv)
- Visuals: SHAP, residuals, and correlation plots in [artifacts/](artifacts)

Run evaluation:

```powershell
python scripts\evaluate_model.py --data data\master_airquality_clean.csv --model models\best_pm25_model.pkl --outdir artifacts
```

### Saving & Loading
- Save best model:

```python
import joblib
joblib.dump(model, "models/best_pm25_model.pkl")
```

- Load for inference (used in API):

```python
model = joblib.load("models/best_pm25_model.pkl")
```

### Reproducibility & Tracking
- Pin dependencies in [requirements.txt](requirements.txt).
- Track experiments (W&B/Comet), code hashes, and data versions.
- Record model metadata (type, features, metrics, date) alongside artifacts.

### Hyperparameter Tuning (Optional)

```powershell
python scripts\hyperparameter_tuning.py --data data\master_airquality_clean.csv --outdir artifacts
```

See detailed methodology and results in [docs/04_MODEL_BUILDING_EVALUATION.md](docs/04_MODEL_BUILDING_EVALUATION.md).

## Model Design Approaches

### Objectives and Constraints
- **Accuracy**: Minimize RMSE/MAE while maintaining generalization.
- **Latency**: Single‑prediction P95 < 100ms for API usage.
- **Interpretability**: Provide explanations via SHAP and importance.
- **Operational fit**: Train on CPU; serve reliably in FastAPI with Prometheus metrics.

### Candidate Models
- **Linear Regression**: Fast, highly interpretable; limited for non‑linear relationships.
- **Random Forest**: Robust to noise; captures non‑linearity; slower inference.
- **XGBoost (Selected)**: Strong performance with regularization; requires tuning; good inference speed.

### Data Split Strategy
- **Temporal split**: Train/Val/Test ≈ 60/20/20 without shuffling to avoid leakage and mimic production (predict future from past).

### Training Configurations
- Linear: Default `LinearRegression()`.
- Random Forest: `n_estimators=100`, `max_depth=20`, `min_samples_split=2`, `random_state=42`.
- XGBoost: `n_estimators=300`, `learning_rate=0.05`, `max_depth=7`, `subsample=0.9`, `colsample_bytree=0.9`, `tree_method='hist'`, `objective='reg:squarederror'`.

### Evaluation & Comparison (Example)
- Metrics: RMSE, MAE, R², MAPE; plots in [artifacts/](artifacts) (pred‑vs‑actual, residuals, error by hour).
- Indicative results: Linear (RMSE ≈ 68), Random Forest (RMSE ≈ 55), XGBoost (RMSE ≈ 52); XGBoost typically best on holdout.

### Interpretability & Error Analysis
- **SHAP**: Global importance emphasizes `PM10`; `CO`/`O3` secondary; temporal features capture periodicity.
- **Failure modes**: Extreme events (festivals, accidents), missing weather context, sensor anomalies.
- **Residual checks**: Inspect heteroscedasticity and time‑of‑day errors.

### Selection Decision & Trade‑offs
- **Chosen**: XGBoost for best accuracy/latency balance and robust regularization.
- **Trade‑offs**: Slightly lower interpretability vs linear baselines; tuning effort required.

### Saving, Loading, and Registry
- Save best model: `models/best_pm25_model.pkl` or specific artifacts like `models/xgb_reg.json`.
- Combine RF chunks: [models/combine_joblib.py](models/combine_joblib.py) → `models/rf_reg.joblib`.
- Load for inference: `joblib.load(...)` in [main.py](main.py); attach Prometheus metrics.
- Optional registry: Manage versions via [scripts/model_registry.py](scripts/model_registry.py).

### Deployment Considerations
- **FastAPI**: Stateless prediction endpoint; compute temporal features from request; log latency and errors.
- **Monitoring**: Use `/metrics` for Prometheus; trigger alerts on latency/error rate thresholds.
- **Fairness**: Evaluate temporal segments; enforce targets in Evaluation Metrics.

### Next Steps
- Hyperparameter tuning (`scripts/hyperparameter_tuning.py`).
- Scheduled retraining and promotion via Kubeflow.
- Expanded features if coverage improves (e.g., weather), with careful missingness handling.

See detailed rationale and results in [docs/04_MODEL_BUILDING_EVALUATION.md](docs/04_MODEL_BUILDING_EVALUATION.md).

## Understanding MLOps

- **Definition**: MLOps blends ML, DevOps, and Data Engineering to reliably deploy and maintain ML systems.
- **Core principles**: Automation, CI/CD, versioning (code/data/models), monitoring, reproducibility, collaboration, and security.
- **Lifecycle**: Data → Model dev → Deployment → Monitoring → Governance.
- **Project mapping**:
   - Data prep: [clean.ipynb](clean.ipynb), [data/](data)
   - Training/eval: [train1.ipynb](train1.ipynb), [scripts/evaluate_model.py](scripts/evaluate_model.py), [artifacts/](artifacts)
   - Orchestration: [kubeflow_pipeline.py](kubeflow_pipeline.py), [pm25_pipeline.yaml](pm25_pipeline.yaml)
   - Serving: [main.py](main.py) (FastAPI), [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)
   - Monitoring: [prometheus.yml](prometheus.yml), [grafana/](grafana), [prometheus/alert_rules.yml](prometheus/alert_rules.yml)
   - Governance: [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), [AUDIT_CHECKLIST.md](AUDIT_CHECKLIST.md), [GOVERNANCE.md](GOVERNANCE.md)

See [docs/02_UNDERSTANDING_MLOPS.md](docs/02_UNDERSTANDING_MLOPS.md) for a deeper guide and maturity model.

## Requirement Analysis

### Functional Requirements
- **Prediction API**: Expose `POST /predict` accepting `PM10`, `O3`, `CO`, `hour`, `dayofweek`, `month`; return PM2.5 prediction. Implemented in [main.py](main.py).
- **Health & Metrics**: `GET /` health and `GET /metrics` Prometheus metrics. Config in [prometheus.yml](prometheus.yml).
- **Data Pipeline**: Ingestion and preprocessing produce [data/master_airquality_clean.csv](data/master_airquality_clean.csv). See [clean.ipynb](clean.ipynb), [scripts/data_preprocessing.py](scripts/data_preprocessing.py).
- **Training & Evaluation**: Train baseline and XGBoost, log metrics/artifacts to [artifacts/](artifacts). See [train1.ipynb](train1.ipynb), [scripts/evaluate_model.py](scripts/evaluate_model.py).
- **Workflow Orchestration**: Compile/deploy pipeline with Kubeflow. See [kubeflow_pipeline.py](kubeflow_pipeline.py), [kubeflow_deploy.py](kubeflow_deploy.py).
- **Drift Monitoring**: Daily or scheduled drift checks; alerts and retraining trigger. See [monitoring/drift_monitor.py](monitoring/drift_monitor.py).
- **Governance Docs**: Maintain [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), [AUDIT_CHECKLIST.md](AUDIT_CHECKLIST.md), [GOVERNANCE.md](GOVERNANCE.md).

### Non-Functional Requirements
- **Latency**: P95 < 100ms for single prediction.
- **Availability**: Uptime > 99.5% with alerting.
- **Reliability**: Error rate < 1%; graceful handling of missing inputs.
- **Scalability**: Horizontal scale via containers.
- **Security**: Restrict env secrets, sanitize inputs; optional auth for protected deployments.
- **Reproducibility**: Deterministic builds; pinned dependencies in [requirements.txt](requirements.txt); tracked runs.
- **Observability**: Prometheus/Grafana dashboards; alert rules in [prometheus/alert_rules.yml](prometheus/alert_rules.yml).

### Acceptance Criteria (Traceable)
- Meets targets in Evaluation Metrics section.
- CI/CD builds and deploys containers; API passes smoke tests.
- Governance artifacts up to date for each model promotion.

## Governance & Compliance

- **Objectives**: Reliability, fairness, transparency, privacy, and clear accountability across the ML lifecycle.
- **Roles**: Data Scientist (model quality/bias), ML Engineer (pipelines/deploy/monitor), Product Owner (requirements & sign‑off), Data Steward (data governance).
- **Processes**:
   - Data governance: Source tracking and licensing in [DATA_CARD.md](DATA_CARD.md); privacy safeguards; access controls.
   - Model development: Experiment tracking (W&B), peer code review, fairness assessment via [governance.ipynb](governance.ipynb).
   - Review & audit: Complete [AUDIT_CHECKLIST.md](AUDIT_CHECKLIST.md); generate and archive governance report ([governance_report.json](governance_report.json)).
   - Deployment & monitoring: Staged rollout, continuous monitoring (Prometheus/Grafana), incident response via alert rules.
- **Artifacts**: Maintain [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), [AUDIT_CHECKLIST.md](AUDIT_CHECKLIST.md), [GOVERNANCE.md](GOVERNANCE.md).
- **Acceptance gates**: Deployment requires meeting Evaluation Metrics (accuracy/latency/uptime), fairness targets (Disparate Impact 0.8–1.25; SPD < 0.1), and completed audit artifacts.

See the full framework in [GOVERNANCE.md](GOVERNANCE.md).

## Data Version Control

### Why DVC
- **Reproducibility**: Track exact data versions used for each model.
- **Traceability**: Link data revisions to experiments and deployments.
- **Efficiency**: Keep large datasets out of Git; store in performant remotes.

### What to Track
- Raw station data: [data/kaggle_csvs/](data/kaggle_csvs)
- Station merges: [data/stations_csvs/](data/stations_csvs)
- Cleaned dataset: [data/master_airquality_clean.csv](data/master_airquality_clean.csv)

### Windows Setup (PowerShell)

```powershell
pip install dvc
dvc init

# Track large/raw datasets and cleaned outputs
dvc add data\kaggle_csvs
dvc add data\stations_csvs
dvc add data\master_airquality_clean.csv

# Commit DVC metadata files
git add data\kaggle_csvs.dvc data\stations_csvs.dvc data\master_airquality_clean.csv.dvc .dvc .gitignore
git commit -m "Track data with DVC"

# Configure a local remote (adjust path if needed)
dvc remote add -d local-storage e:\5th SEM Data\AI254TA-Machine Learning Operations(MLOps)\MLOPs_Project\dvc_storage
dvc push
```

### Daily Workflows
- **Pull data for experiments**:

```powershell
dvc pull
```

- **Update tracked datasets after pipeline runs**:

```powershell
python scripts\data_ingestion.py
python scripts\data_preprocessing.py
dvc add data\master_airquality_clean.csv
git commit -m "Update cleaned dataset version"
dvc push
```

### Policies & Tips
- Do not commit raw CSVs to Git; rely on `.dvc` files and remotes.
- Pin preprocessing code and record hashes with experiment tracking.
- Ensure CI agents can `dvc pull` using an accessible remote (local/SSH/cloud).

See repository storage in [dvc_storage/](dvc_storage/) and governance policies in [GOVERNANCE.md](GOVERNANCE.md).

## Risks and Risk Matrix

- **Methodology**: 5×5 Risk Matrix (Likelihood 1–5 × Impact 1–5). See [RISK_ASSESSMENT.md](RISK_ASSESSMENT.md) and [risk_matrix_5x5.html](risk_matrix_5x5.html).

### Key Risks (Examples)
- **Data Drift (Score ≈ 16, High)**: Seasonal shifts or new sources alter `PM10`, `O3`, `CO` distributions.
   - Mitigation: Daily KS/PSI checks; Evidently reports; monthly retraining per [RETRAINING_PLAN.md](RETRAINING_PLAN.md); alerts on thresholds.
- **Model Bias (Score ≈ 12, High)**: Uneven performance across time segments (weekday/weekend).
   - Mitigation: AIF360 audits; enforce Disparate Impact 0.8–1.25; document in [MODEL_CARD.md](MODEL_CARD.md).
- **Latency Spikes (Score ≈ 9, Medium)**: P95 > 100ms under load.
   - Mitigation: Scale replicas; optimize model; cache hot requests; monitor via Prometheus.
- **Sensor Failures (Score ≈ 6, Medium)**: Missing upstream data.
   - Mitigation: Input validation and imputation; fail-safe responses.
- **Config/Dependency Drift (Variable)**: Upgrades break compatibility.
   - Mitigation: Pin dependencies; CI checks; canary deployments.

### Ownership & Playbooks
- **Owners**: Data Scientist (model quality/bias), ML Engineer (infra/monitoring), Data Steward (data governance).
- **Triggers**: Alert rules in [prometheus/alert_rules.yml](prometheus/alert_rules.yml).
- **Actions**: Run [scripts/evaluate_model.py](scripts/evaluate_model.py), review [artifacts/](artifacts), initiate retrain via Kubeflow, update governance docs.

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
