
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
# Data Card: Urban Air Quality Dataset

## Dataset Overview
This dataset aggregates air quality measurements from multiple cities and stations to predict PM2.5 levels. It combines historical pollutant data with temporal features.

## Dataset Metadata
- **Name**: Master Air Quality Clean Dataset
- **Source**: Aggregated from various city-specific CSVs (e.g., AP001, DL001) located in `data/kaggle_csvs/`.
- **Version**: 1.0
- **License**: [Insert License Here, e.g., CC BY 4.0 or Open Database License]
- **Maintainers**: MLOps Project Team

## Data Composition
- **Total Records**: ~12 million (based on governance report)
- **Time Range**: [Start Date] to [End Date]
- **Granularity**: Hourly measurements

## Features
| Feature | Type | Description |
|---------|------|-------------|
| `PM2.5` | Float | Target variable. Particulate Matter < 2.5 micrometers. |
| `PM10` | Float | Particulate Matter < 10 micrometers. |
| `O3` | Float | Ozone concentration. |
| `CO` | Float | Carbon Monoxide concentration. |
| `Timestamp` | DateTime | Date and time of measurement. |
| `hour` | Integer | Hour of the day (0-23). |
| `dayofweek` | Integer | Day of the week (0=Monday, 6=Sunday). |
| `month` | Integer | Month of the year (1-12). |

## Data Cleaning & Preprocessing
1. **Missing Values**: Rows with missing `Timestamp` or `PM2.5` are dropped. Other features (`PM10`, `O3`, `CO`) are imputed using the median.
2. **Type Conversion**: Pollutant columns are converted to numeric, coercing errors to NaN.
3. **Feature Engineering**: Temporal features (`hour`, `dayofweek`, `month`) are extracted from `Timestamp`.

## Known Limitations & Biases
- **Geographic Bias**: Data may be heavily weighted towards specific cities (e.g., Delhi) depending on the number of stations.
- **Temporal Gaps**: Some stations may have missing data for extended periods.
- **Sensor Error**: Raw sensor data may contain outliers or calibration errors.

## Governance & Compliance
- **PII**: No Personally Identifiable Information is contained in this dataset.
- **Fairness**: Evaluated for bias across temporal groups (e.g., Weekend vs. Weekday). See `governance_report.json`.
