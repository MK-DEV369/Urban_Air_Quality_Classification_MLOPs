
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
# Risk Assessment: Air Quality Prediction System

## 1. Methodology
We utilize a standard **5x5 Risk Matrix** to evaluate risks based on two dimensions:
- **Likelihood**: Rare (1) to Almost Certain (5)
- **Impact**: Insignificant (1) to Catastrophic (5)

**Risk Score = Likelihood × Impact**
- **Low (1-4)**: Acceptable risk.
- **Medium (5-9)**: Monitor and mitigate.
- **High (10-19)**: Urgent action required.
- **Extreme (20-25)**: Stop deployment until resolved.

## 2. Top Risks Identified

### 2.1 Data Drift (Score: 16 - High)
- **Description**: The distribution of input features (e.g., PM10, O3) changes over time due to seasonal shifts or new pollution sources.
- **Likelihood**: 4 (Likely)
- **Impact**: 4 (Major - Model accuracy degrades significantly)
- **Mitigation**:
    - Implement automated drift detection using Evidently AI.
    - Schedule monthly retraining.
    - Alert when drift score > 0.1.

### 2.2 Model Bias (Score: 12 - High)
- **Description**: The model performs significantly worse for specific time periods (e.g., weekends) or locations.
- **Likelihood**: 3 (Possible)
- **Impact**: 4 (Major - Reputational damage and poor decision making)
- **Mitigation**:
    - Conduct fairness audit using AIF360 (Disparate Impact analysis).
    - Include fairness metrics in the model card.
    - Reject models with Disparate Impact < 0.8 during evaluation.

### 2.3 API Latency Spikes (Score: 9 - Medium)
- **Description**: Inference time exceeds 500ms during high load.
- **Likelihood**: 3 (Possible)
- **Impact**: 3 (Moderate - Poor user experience)
- **Mitigation**:
    - Horizontal scaling of FastAPI containers (Kubernetes/Docker Swarm).
    - Cache frequent predictions.
    - Monitor P95 latency via Prometheus.

### 2.4 Sensor Failure / Missing Data (Score: 6 - Medium)
- **Description**: Upstream sensors fail, leading to missing values in input.
- **Likelihood**: 3 (Possible)
- **Impact**: 2 (Minor - Model can handle some missingness)
- **Mitigation**:
    - Robust imputation strategy (median filling) in the inference pipeline.
    - Data quality checks before inference.

## 3. Risk Matrix Visualization
See `risk_matrix_5x5.html` for an interactive visualization of these risks.
