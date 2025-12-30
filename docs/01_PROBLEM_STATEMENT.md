# Problem Statement and Project Objectives

## 1. Problem Statement

Urban air pollution, particularly PM2.5 (Particulate Matter less than 2.5 micrometers), poses significant health risks to populations worldwide. Real-time prediction of PM2.5 levels can help:
- Public health officials issue timely warnings
- Individuals make informed decisions about outdoor activities
- City planners implement pollution control measures

**Challenge**: Build a robust, production-ready machine learning system that predicts PM2.5 levels based on other pollutants and temporal features.

## 2. Project Objectives

### Primary Objectives
1. **Develop an ML model** that predicts PM2.5 concentrations with RMSE < 30 µg/m³
2. **Deploy a production API** that serves real-time predictions with <100ms latency
3. **Implement MLOps practices** for continuous model monitoring and improvement

### Secondary Objectives
1. Ensure model fairness across different time periods (weekday/weekend)
2. Maintain model interpretability through SHAP analysis
3. Automate retraining pipelines triggered by data drift
4. Achieve 99.5% API uptime

## 3. Project Scope

### In Scope


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
- Forecasting future pollution levels (time series prediction)
- Causal inference (what causes pollution)
- Multi-city deployment (limited to Indian cities in dataset)
- Mobile application development

## 4. Evaluation Metrics

### Model Performance Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| **RMSE** | < 30 µg/m³ | Primary metric - penalizes large errors |
| **MAE** | < 20 µg/m³ | Average absolute error for interpretability |
| **R²** | > 0.75 | Proportion of variance explained |
| **MAPE** | < 40% | Percentage error for business context |

### Operational Metrics
| Metric | Target | Tool |
|--------|--------|------|
| **API Latency (P95)** | < 100ms | Prometheus |
| **API Uptime** | > 99.5% | Prometheus |
| **Error Rate** | < 1% | Prometheus/Grafana |
| **Drift Detection** | Monitored daily | Custom KS/PSI |

### Fairness Metrics
| Metric | Target | Tool |
|--------|--------|------|
| **Disparate Impact** | 0.8 - 1.25 | AIF360 |
| **Statistical Parity** | < 0.1 | AIF360 |

## 5. Success Criteria

The project is considered successful if:
1. ✅ Model achieves target performance metrics on holdout test set
2. ✅ API serves predictions with acceptable latency
3. ✅ CI/CD pipeline automates training and deployment
4. ✅ Monitoring system detects drift and triggers alerts
5. ✅ All governance documentation is complete and signed off
6. ✅ Model passes fairness evaluation

## 6. Stakeholders

| Role | Responsibility |
|------|----------------|
| **Data Scientist** | Model development, evaluation, fairness analysis |
| **ML Engineer** | Pipeline automation, deployment, monitoring |
| **Product Owner** | Requirements, acceptance criteria, business value |
| **Data Steward** | Data governance, privacy, compliance |

## 7. Timeline (Indicative)

- **Week 1-2**: Data exploration, cleaning, feature engineering
- **Week 3-4**: Model training, evaluation, hyperparameter tuning
- **Week 5**: CI/CD pipeline setup, Docker containerization
- **Week 6**: Deployment, monitoring setup, governance documentation
- **Week 7+**: Continuous monitoring, retraining, improvements
