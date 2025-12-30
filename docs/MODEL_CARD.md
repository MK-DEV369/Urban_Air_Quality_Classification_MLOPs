
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
# Model Card: PM2.5 Prediction Model

## Model Details
- **Model name**: best_pm25_model.pkl
- **Model type**: (auto-filled at runtime) e.g., XGBRegressor
- **Version**: v1
- **Framework**: scikit-learn / xgboost
- **Location**: `models/best_pm25_model.pkl`

## Intended Use
- Predict PM2.5 concentration using measured pollutants + time features.

## Inputs
Expected features used by the API:
- `PM10` (float)
- `O3` (float)
- `CO` (float)
- `hour` (int)
- `dayofweek` (int)
- `month` (int)

## Output
- `PM25_prediction` (float)

## Training Data
- Reference dataset: `data/master_airquality_clean.csv`
- Preprocessing: performed in `clean.ipynb` (missing values handling + time features)

## Evaluation
- Metrics reported in: W&B runs and training notebook (`train1.ipynb`)

## Ethical Considerations
- Bias/fairness analysis: `governance.ipynb`
- Governance report: `governance_report.json`

## Limitations
- Model quality depends on sensor coverage and data quality.
- Temporal/seasonal drift may degrade performance.

## Monitoring
- Operational metrics: `/metrics` (Prometheus)
- Drift checks: `monitoring/evidently_drift_report.py` (optional)
