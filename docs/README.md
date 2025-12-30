
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
# Revisiting Model Performance

A practical guide for reassessing the PM2.5 prediction model after deployment or significant data/model changes.

## When to Revisit
- Drift alerts: Feature/target drift (KS/PSI) or sudden distribution shifts.
- Performance regression: RMSE/MAE/R²/MAPE worse than acceptance targets or prior release.
- Operational issues: Latency/uptime SLO breaches, elevated error rates.
- Fairness signals: Temporal segment gaps (weekday/weekend), or DI/SPD outside targets.
- Data updates: New stations/cities ingested, major seasonal changes, or sensor schema changes.
- Release cycle: Scheduled retraining or major dependency upgrades.

## Acceptance Targets (Holdout)
- RMSE < 30 µg/m³
- MAE < 20 µg/m³
- R² > 0.75
- MAPE < 40%
- P95 latency < 100ms; uptime > 99.5%; error rate < 1%
- Fairness: Disparate Impact 0.8–1.25; Statistical Parity Difference < 0.1

## Checklist (Fast Pass)
1. Pull latest data/model artifacts (`dvc pull` if enabled).
2. Run evaluation + plots:
   - `python scripts/evaluate_model.py --data data/master_airquality_clean.csv --model models/best_pm25_model.pkl --outdir artifacts`
   - `python scripts/shap_analysis.py --model models/best_pm25_model.pkl --data data/master_airquality_clean.csv --outdir artifacts`
3. Compare metrics to targets and last promoted model (use experiment tracking run history).
4. Inspect visuals: residuals, predicted vs actual, error by hour, SHAP summary/dependence.
5. Check segment performance (hour/day/month; stations if available) and fairness metrics.
6. Confirm latency/uptime/error-rate SLOs from Prometheus/Grafana.
7. Document findings in experiment tracking with data/code hashes.

## Deep-Dive Steps
- **Residuals**: Look for skew or funnel shapes; investigate outliers via `artifacts/worst_predictions.csv`.
- **Segments**: Quantify MAE/RMSE by hour/day/month; flag segments exceeding acceptable deltas.
- **Drift**: Compare recent batch feature distributions to training (KS/PSI); note shifts in PM10/CO/O3 and temporal segments.
- **Fairness**: Recompute DI/SPD across temporal segments; note any regressions.
- **Calibration**: Check predicted vs actual for systematic offsets; consider calibration or feature augmentation.

## Actions Based on Findings
- **High rush-hour errors**: Add congestion/weather proxies; retrain with updated features.
- **Outlier-driven RMSE**: Review sensor anomalies; adjust outlier handling or robust loss.
- **Segment bias**: Reweight or augment data; tighten fairness gates before promotion.
- **Latency regressions**: Optimize model (tree depth/rounds), or scale replicas; cache frequent requests.
- **Drift detected**: Trigger retraining pipeline; update data version; rerun evaluation.

## Promotion Criteria
- Meets/recovers acceptance targets vs prior release.
- Fairness within target bands; no new critical segment regressions.
- Latency/uptime/error-rate within SLOs in staging.
- All artifacts logged (metrics, plots, predictions, SHAP) with data/code hashes.
- Governance updated: MODEL_CARD, AUDIT_CHECKLIST, governance report.

## Traceability & Logging
- Record: data version (DVC hash/date tag), model file hash, git commit, env (Python + key libs).
- Store artifacts in `artifacts/` and log to experiment tracker (Comet/W&B).
- Note manual exclusions or notable events (festivals, sensor outages) in run notes.

## Useful Commands (PowerShell)
```powershell
# Evaluate current model
python scripts\evaluate_model.py --data data\master_airquality_clean.csv --model models\best_pm25_model.pkl --outdir artifacts

# SHAP analysis
python scripts\shap_analysis.py --model models\best_pm25_model.pkl --data data\master_airquality_clean.csv --outdir artifacts

# Pull data via DVC (if configured)
dvc pull
```

## Related References
- docs/04_MODEL_BUILDING_EVALUATION.md (metrics, visualization, tracking, interpretation)
- README.md (objectives, metrics, monitoring)
- GOVERNANCE.md (acceptance gates, audit artifacts)
