
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
# AI Governance Framework

This document outlines the governance framework for the Air Quality Prediction project. It defines the roles, responsibilities, and processes to ensure the responsible development, deployment, and monitoring of our AI models.

## 1. Objectives

- Ensure model reliability, fairness, and transparency.
- Maintain compliance with data privacy and security standards.
- Establish clear accountability for model lifecycle stages.
- Mitigate risks associated with AI deployment.

## 2. Roles & Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Data Scientist** | Model development, feature engineering, fairness evaluation, maintaining the Model Card. |
| **ML Engineer** | Pipeline automation (CI/CD), model deployment, monitoring setup, infrastructure management. |
| **Product Owner** | Defining business requirements, acceptance criteria, and final sign-off for deployment. |
| **Data Steward** | Data quality assurance, access control, and compliance with data licensing/privacy. |

## 3. Governance Process

### 3.1 Data Governance
- **Source Tracking**: All data sources must be documented in `README.md` or `DATA_CARD.md`.
- **Versioning**: Raw and processed data must be versioned using DVC.
- **Privacy**: PII must be removed or anonymized before training.

### 3.2 Model Development
- **Experiment Tracking**: All experiments must be tracked (e.g., using MLflow or Weights & Biases) to ensure reproducibility.
- **Code Review**: All code changes require a pull request review by at least one peer.
- **Fairness Assessment**: Models must be evaluated for bias using the `governance.ipynb` workflow before promotion.

### 3.3 Model Review & Audit
- Before deployment to production, the **Audit Checklist** (`AUDIT_CHECKLIST.md`) must be completed.
- A "Governance Report" (e.g., `governance_report.json`) must be generated and reviewed.

### 3.4 Deployment & Monitoring
- **Staged Deployment**: Models are first deployed to a staging environment for integration testing.
- **Continuous Monitoring**: Production models are monitored for drift and performance degradation.
- **Incident Response**: Alerts triggered by the monitoring system must be addressed within the defined SLA.

## 4. Risk Management

Refer to the Risk Matrix (e.g., `risk_matrix_5x5.html`) for identified risks and mitigation strategies.

## 5. Documentation Standards

- **Model Card**: Must be updated for every major version.
- **Audit Checklist**: Must be signed off for every production release.
- **Governance Report**: Must be archived with the model artifacts.
