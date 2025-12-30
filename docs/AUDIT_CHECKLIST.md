
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
# Model Audit & Review Checklist

This document serves as a formal checklist for auditing and reviewing machine learning models before deployment and during their lifecycle.

## 1. Data Governance & Quality

- [ ] **Data Lineage**: Is the origin of the training data documented? (Source, collection method, date range)
- [ ] **Data Versioning**: Is the dataset versioned (e.g., using DVC)?
- [ ] **Data Privacy**: Have PII (Personally Identifiable Information) checks been performed?
- [ ] **Bias Check**: Has the data been checked for representation bias across key demographics or groups?
- [ ] **Data Quality**: Are there checks for missing values, outliers, and schema validation?
- [ ] **Licensing**: Is the data usage compliant with its license?

## 2. Model Development & Training

- [ ] **Reproducibility**: Can the model training be reproduced from the code and data version? (Seed setting, environment specification)
- [ ] **Algorithm Selection**: Is the choice of algorithm justified and documented?
- [ ] **Hyperparameter Tuning**: Is the tuning process and selected hyperparameters documented?
- [ ] **Code Quality**: Has the training code been peer-reviewed?
- [ ] **Environment**: Are all dependencies listed in `requirements.txt` or `environment.yml` with pinned versions?

## 3. Model Evaluation

- [ ] **Metrics**: Are appropriate evaluation metrics defined (e.g., RMSE, MAE, R2) and aligned with business objectives?
- [ ] **Test Set**: Is the test set separate and representative of production data?
- [ ] **Baseline Comparison**: Is the model performance compared against a simple baseline or previous version?
- [ ] **Error Analysis**: Has an analysis of error cases (worst predictions) been performed?
- [ ] **Fairness**: Have fairness metrics (e.g., Disparate Impact, Equal Opportunity) been calculated? (See `governance_report.json`)
- [ ] **Overfitting**: Is there a check for overfitting (Train vs. Validation performance)?

## 4. Model Artifacts & Documentation

- [ ] **Model Card**: Is the `MODEL_CARD.md` up-to-date with current model details?
- [ ] **Artifact Storage**: Is the model artifact stored in a secure and versioned registry (e.g., MLflow, S3)?
- [ ] **Input/Output Schema**: Are the expected input features and output format clearly defined?
- [ ] **Interpretability**: Is there an explanation of feature importance or SHAP values?

## 5. Deployment & Operations (MLOps)

- [ ] **CI/CD**: Does the deployment pipeline pass all automated tests?
- [ ] **Containerization**: Is the Docker image built and scanned for vulnerabilities?
- [ ] **Scalability**: Has the inference service been load-tested?
- [ ] **Rollback Plan**: Is there a documented procedure to roll back to a previous model version?

## 6. Monitoring & Maintenance

- [ ] **Drift Detection**: Is monitoring set up for data drift and concept drift?
- [ ] **Performance Monitoring**: Are operational metrics (latency, error rate, CPU/Memory) being tracked (e.g., Prometheus/Grafana)?
- [ ] **Alerting**: Are alerts configured for critical failures or performance degradation?
- [ ] **Retraining Policy**: Is the criteria and process for retraining defined?

## 7. Sign-off

- [ ] **Data Scientist**: ____________________ Date: __________
- [ ] **ML Engineer**: ____________________ Date: __________
- [ ] **Product Owner**: ____________________ Date: __________
