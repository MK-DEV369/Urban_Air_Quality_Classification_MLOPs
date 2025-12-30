# Chapter 10: End-to-End MLOps Pipeline Demo, Evaluation, and Final Deliverables

Comprehensive walkthrough of running the full PM2.5 MLOps pipeline, evaluating final results, and assembling project documentation, reporting, and presentation.

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

## End-to-End MLOps Pipeline Demonstration
- **Scope**: Run data ingestion → preprocessing → training → evaluation → packaging → deployment → monitoring.
- **Entry points**: `scripts/pipeline.py` (local), `kubeflow_pipeline.py` / `pm25_pipeline.yaml` (Kubeflow), or CI workflow (`.github/workflows/mlops.yml`).
- **Data**: Use `data/master_airquality_clean.csv` (or DVC remote) as reference; optionally sample for fast demo.
- **Steps**:
  1) Ingest/clean: `scripts/data_ingestion.py`, `scripts/data_preprocessing.py` to produce cleaned features.
  2) Train: `scripts/train_with_comet.py` to train RF/XGB; log run IDs and params.
  3) Evaluate: `scripts/evaluate_model.py` to generate `artifacts/evaluation_metrics.json`, predictions, SHAP plots.
  4) Register: `scripts/model_registry.py` to store candidate with data hash, code commit, metrics.
  5) Package: build Docker image (Dockerfile) embedding the approved model artifact.
  6) Deploy: `docker-compose.yml` (or Kubeflow deploy) to staging; health check `/health` and `/metrics`.
  7) Monitor: Prometheus/Grafana for latency/error/drift; Evidently/KS/PSI via `monitoring/drift_monitor.py`.
- **Validation in staging**: run smoke predict with a canned payload; confirm model/version labels in metrics.

## Final Model Evaluation and Results
- **Primary metrics**: RMSE, MAE, R², MAPE on holdout; include confidence intervals if available.
- **Segment metrics**: evaluate by station/region/time-of-day to uncover localized regressions.
- **Fairness**: report disparate impact or mean error gaps across key groups; target DI 0.8–1.25, SPD < 0.1.
- **Latency and reliability**: P95 latency, error rate, uptime during evaluation window.
- **Artifacts**: consolidate `artifacts/evaluation_metrics.json`, predictions CSVs, SHAP plots, and drift reports (Evidently HTML/JSON).
- **Decision**: promote, retrain, or rollback; document rationale, thresholds, and comparisons to previous model.

## Project Documentation and Reporting
- **Core docs**: ensure `MODEL_CARD.md`, `DATA_CARD.md`, `GOVERNANCE.md`, `RETRAINING_PLAN.md`, `AUDIT_CHECKLIST.md` are updated with the final run.
- **Changelogs**: note model version, data hash (DVC), code commit, and deployment tag.
- **Run lineage**: link experiment tracker run ID (Comet/W&B) to registry entry and deployed image tag.
- **Dashboards**: export key Grafana panels (API performance, drift) as PDFs/PNGs for the report.
- **Risk/mitigations**: summarize outstanding risks and controls (rate limits, input validation, rollback).

## Final Presentation and Evaluation
- **Audience**: blend technical (pipeline, metrics, architecture) and stakeholder value (reliability, risk controls).
- **Story arc**: problem → data → model → pipeline → deployment → monitoring → governance → results → next steps.
- **Live demo**: optional short demo calling `/predict` and showing live Grafana metrics; keep rollback ready.
- **Acceptance gates**: clearly state thresholds and whether they are met (metrics, fairness, latency, error rate).
- **Next steps**: roadmap for improvements (feature updates, retraining cadence, alert tuning, cost/perf optimization).

## Audit and Review Checklist (Final Pass)
- Performance: metrics vs. baseline and SLOs; segment/fairness checks.
- Drift: PSI/KS status; concept drift via residual trends.
- Observability: Prometheus/Grafana dashboards active; alerts tested.
- Security/Privacy: secrets management, input validation, log hygiene confirmed.
- Documentation: Model/Data Cards, governance, audit checklist updated.
- Rollback readiness: N−1 image available; playbook rehearsed.
- Approvals: promotion signed off; evidence archived.

## Project File Pointers
- Pipelines and orchestration: `scripts/pipeline.py`, `kubeflow_pipeline.py`, `pm25_pipeline.yaml`
- Training/eval: `scripts/train_with_comet.py`, `scripts/evaluate_model.py`
- Registry/promotion: `scripts/model_registry.py`
- Deployment: `Dockerfile`, `docker-compose.yml`, `kubeflow_deploy.py`
- Monitoring: `main.py` (metrics), `prometheus.yml`, `prometheus/alert_rules.yml`, `monitoring/drift_monitor.py`
- Documentation: `MODEL_CARD.md`, `DATA_CARD.md`, `GOVERNANCE.md`, `AUDIT_CHECKLIST.md`, `RETRAINING_PLAN.md`
