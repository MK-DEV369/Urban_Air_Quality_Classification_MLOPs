
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
# Chapter 8: Monitoring, Drift, and Redeployment

A project-specific guide to monitoring the PM2.5 model and service, detecting drift, and executing retraining/redeployment with Prometheus, Grafana, and EvidentlyAI.

## 1) Model Monitoring Concepts and KPIs
- **Service health**: uptime (`up`), instance availability.
- **Performance**: request throughput (`api_requests_total`), latency (`request_latency_seconds` P50/P95/P99).
- **Quality KPIs** (from `artifacts/evaluation_metrics.json` and online checks): RMSE, MAE, R², MAPE; segment/fairness deltas if available.
- **Reliability**: error rate (5xx fraction), timeout rate, queue depth (if applicable).
- **Business proxies**: % predictions above/below regulatory thresholds; optional SLA adherence.
- **Model freshness**: model version/commit tag, data version (DVC hash), model age since last promotion.

## 2) Tools and Roles
- **Prometheus**: scrape `/metrics` from FastAPI (`main.py`), store time-series, evaluate alert rules in `prometheus/alert_rules.yml`.
- **Grafana**: visualize latency/throughput/error rate and drift dashboards; datasource: Prometheus at `http://prometheus:9090`.
- **EvidentlyAI**: offline/online drift reports; use in `monitoring/drift_monitor.py` or notebook to compare reference vs current batches.

## 3) Detecting Concept Drift and Data Drift
- **Data drift**: feature distribution shifts. Use KS test and PSI (already in `monitoring/drift_monitor.py`); set thresholds (PSI <0.1 stable, 0.1–0.2 moderate, >0.2 significant).
- **Concept drift**: target/prediction relationship changes. Track prediction error over time; compare live errors against reference window. Use Evidently regression report to monitor error distribution, residual mean, and correlation changes.
- **Pipelines**:
  - **Batch drift job** (cron or Actions schedule): load reference slice (e.g., last stable week) and current slice (latest day), run Evidently report, write JSON/HTML to `monitoring/reports/`, and emit summary metrics to Prometheus (via pushgateway or file export for scrape).
  - **Online drift signals**: lightweight KS/PSI on recent predictions stream (sampled) or feature stats logged from `main.py` middleware.
- **Alerts for drift**:
  - Trigger when PSI > 0.2 or KS p-value < 0.05 for key features.
  - Trigger when rolling RMSE/MAE exceeds reference by >10% for 3 consecutive windows.
  - Optional: alert on feature missingness/spike in NaNs.

## 4) Model Retraining and Redeployment
- **Triggers**: drift alert breach, performance regression (RMSE/MAE/MAPE/R² drop), model freshness threshold, or scheduled cadence.
- **Retraining flow**:
  1. Pull latest data (`data/master_airquality_clean.csv` or DVC remote).
  2. Run `scripts/train_with_comet.py` (or `kubeflow_pipeline.py`) to train; log run ID, params, metrics.
  3. Evaluate with `scripts/evaluate_model.py`; store `artifacts/evaluation_metrics.json`, SHAP plots, and predictions.
  4. Register candidate in `scripts/model_registry.py` with metadata (data hash, commit, run ID).
- **Promotion decision**: accept if KPIs improve and fairness/latency budgets are met; record in `MODEL_CARD.md` and registry.
- **Redeployment**:
  - Build image (Dockerfile) embedding the approved model artifact; tag with commit SHA and model version.
  - Deploy to staging via Compose/Kubernetes; run smoke (`/health`, `/metrics`, sample predict) and short shadow/AB if possible.
  - Promote the exact image tag to production; keep N−1 for rollback.

## 5) Logging, Alerting, and Dashboard Setup
- **Logging**: FastAPI logs request/response status and timing; ensure prediction payload size is bounded; log model version and request ID for tracing.
- **Metrics exposure**: `/metrics` already emits counters/histograms for requests and latency. Extend to include `model_version_info` gauge and drift job results (counts of features in drift).
- **Alerting** (`prometheus/alert_rules.yml` examples):
  - Latency: P95 > 100ms for 5m.
  - Error rate: 5xx > 5% for 1m.
  - Instance down: `up == 0` for 1m.
  - Drift: custom metric `drift_features_total > 0`.
- **Dashboards** (Grafana):
  - API performance: throughput, latency P95, error rate, active alerts.
  - Model quality: rolling RMSE/MAE from eval artifacts pushed as Prometheus gauges or uploaded via pushgateway.
  - Drift: feature-level PSI/KS visualized from Evidently outputs (export Prometheus metrics or load JSON into Grafana using JSON data source).
- **Alert delivery**: configure Alertmanager routes (email/Slack/Teams); include run/model version, affected feature, and suggested action (retrain or rollback).

## 6) Operational Runbook (quick steps)
- Daily/cron: run `monitoring/drift_monitor.py`; review `monitoring/reports/` and Grafana drift panels.
- If drift/regression fires: confirm data pipeline health; trigger retraining workflow; compare candidate vs production metrics; decide promote/rollback.
- After redeploy: watch `/metrics` for error/latency, validate one prediction, and monitor drift counters reset.

## 7) Project File Pointers
- Service + metrics: `main.py`
- Alerts and Prometheus config: `prometheus.yml`, `prometheus/alert_rules.yml`
- Drift jobs: `monitoring/drift_monitor.py`, reports in `monitoring/reports/`
- Retraining/eval: `scripts/train_with_comet.py`, `scripts/evaluate_model.py`
- Registry/promotion: `scripts/model_registry.py`
- Deployment: `docker-compose.yml`, `kubeflow_deploy.py`, `kubeflow_pipeline.py`
- Artifacts/metrics: `artifacts/` (evaluation_metrics.json, predictions/shap plots)
