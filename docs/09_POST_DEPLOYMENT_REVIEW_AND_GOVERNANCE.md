# Chapter 9: Post-Deployment Review, Feedback Loops, and Governance

Comprehensive guidance on evaluating the PM2.5 model after deployment, closing feedback loops, and maintaining ethical, documented, and auditable ML operations.

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

---

## Post-Deployment Performance Review
- **What to assess**: online latency/error rate, rolling RMSE/MAE/MAPE/R² (from eval jobs), drift signals (PSI/KS), freshness (model age, data hash), fairness gaps across key segments, and incident history since last promotion.
- **Windows**: use short windows (hour/day) for responsiveness and longer windows (week/month) for stability; compare to reference baselines captured at promotion.
- **Evidence package**: collect metrics from `artifacts/evaluation_metrics.json`, Prometheus (latency/error/throughput), Grafana dashboards, and drift reports in `monitoring/reports/`.
- **Decision outcomes**: continue, retrain, rollback to N−1, or hotfix (e.g., rule-based guardrails).
- **Runbook**: document the review in the changelog with model version, data hash, and decision.

## Continuous Improvement via Feedback Loops
- **Feedback sources**: monitored errors, user reports, domain SME input, and periodic drift checks.
- **Loop design**: capture signals → triage (noise vs. actionable) → plan (retrain/tune/features) → execute → validate in staging → promote or reject.
- **Data curation**: add mislabeled/outlier cases and hard negatives to the training set; maintain a curated incremental dataset.
- **Experimentation**: run controlled comparisons (A/B or shadow) when deploying a candidate; require statistically or operationally meaningful improvements.
- **Closing the loop**: update MODEL_CARD and registry entry with the outcome; log learnings to avoid repeated regressions.

## Governance and Ethical AI Practices
- **Fairness monitoring**: track error parity and outcome parity across key groups (e.g., regions/stations/time-of-day). Set thresholds (e.g., DI 0.8–1.25, SPD < 0.1).
- **Transparency**: keep MODEL_CARD and DATA_CARD current at each promotion with data sources, known limits, and caveats.
- **Accountability**: maintain approval steps for promotion; require two-person review for high-impact changes.
- **Privacy and security**: minimize logged PII; secure secrets; validate inputs (schema, ranges) to prevent abuse.
- **Compliance**: ensure retraining and deployment follow documented SOPs; retain audit artifacts.

## Documentation: Model Cards and Data Cards
- **Model Card updates**: record version, data slice used, metrics (overall and by segment), calibration or drift notes, and known failure modes. File: `MODEL_CARD.md`.
- **Data Card updates**: document sources, recency, schema changes, quality notes, and known biases. File: `DATA_CARD.md`.
- **Linkage**: cross-reference model version to data hash (DVC) and code commit; store in registry entries.
- **Review cadence**: update both cards on every promotion and after material drift/retraining events.

## Audit and Review Checklist
- **Performance**: latest eval metrics vs. baseline; latency/error SLO adherence.
- **Drift**: PSI/KS status for key features; concept drift via residual trends.
- **Fairness**: segment metrics within agreed thresholds; note mitigations if not.
- **Freshness**: model age, data vintage, and last promotion date.
- **Security/Privacy**: secrets managed, inputs validated, logs sanitized.
- **Observability**: metrics, logs, dashboards, and alerts in place and tested.
- **Documentation**: MODEL_CARD, DATA_CARD, AUDIT_CHECKLIST updated; registry entry present.
- **Rollback readiness**: N−1 image available; rollback playbook verified.
- **Approvals**: promotion reviewed and signed off per policy.

## Project File Pointers
- Metrics and service: `main.py`, `prometheus.yml`, `prometheus/alert_rules.yml`
- Drift and reports: `monitoring/drift_monitor.py`, `monitoring/reports/`
- Evaluation artifacts: `artifacts/evaluation_metrics.json`, predictions/SHAP plots
- Documentation: `MODEL_CARD.md`, `DATA_CARD.md`, `AUDIT_CHECKLIST.md`, `GOVERNANCE.md`
- Registry and promotion: `scripts/model_registry.py`
- Deployment and rollback: `docker-compose.yml`, `kubeflow_deploy.py`, `kubeflow_pipeline.py`
