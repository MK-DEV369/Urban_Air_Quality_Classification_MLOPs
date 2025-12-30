
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
# Model Retraining & Redeployment Plan

## 1. Retraining Strategy

### 1.1 Triggers
Retraining is triggered by one of the following events:
- **Scheduled**: Monthly retraining to incorporate new data.
- **Performance Drift**: If model performance (RMSE) degrades by >10% compared to the baseline on the monitoring dashboard.
- **Data Drift**: If the drift score (calculated via Evidently AI) for key features (`PM10`, `O3`) exceeds the threshold (0.1).

### 1.2 Data Window
- **Rolling Window**: Train on the last 12 months of data to capture seasonal trends while remaining current.
- **Validation**: Use the most recent 1 month for validation.

## 2. Evaluation Gates

Before a new model version is promoted to production, it must pass the following gates:
1.  **Metric Improvement**: The new model must show equal or better RMSE/MAE than the currently deployed model on the holdout test set.
2.  **Fairness Check**: The Disparate Impact ratio must remain within the range [0.8, 1.25].
3.  **Latency Check**: Inference time must not exceed 100ms (P95).

## 3. Redeployment Process

### 3.1 Continuous Deployment (CD) Pipeline
1.  **Build**: A new Docker image is built with the updated model artifact.
2.  **Test**: Automated unit and integration tests run.
3.  **Staging**: Deploy to a staging environment.
4.  **Approval**: Manual or automated sign-off based on evaluation gates.
5.  **Production**: Rolling update to production (Kubernetes/Docker Compose).

### 3.2 Rollback
If the error rate spikes >5% post-deployment:
1.  Revert to the previous Docker image tag.
2.  Mark the new model version as "Rejected" in the model registry.

## 4. Feedback Loop
- **Ground Truth Collection**: Actual PM2.5 values are collected from reference stations daily.
- **Residual Analysis**: Residuals (Predicted - Actual) are analyzed to identify systematic errors (e.g., underprediction during holidays).
