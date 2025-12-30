
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
# Overview of CI/CD for ML (Chapter 7)

A detailed guide to the CI/CD approach used in this project for ML models and the FastAPI service.

## Objectives
- Automate build, test, and quality checks for code and models.
- Ensure reproducible training/evaluation with tracked artifacts.
- Provide safe, auditable deployment to staging and production.
- Enable rapid rollback if regressions occur.

## Pipeline Stages (Typical)
1. **CI: Lint & Unit Checks**
   - Install `requirements.txt`.
   - Lint/format check (e.g., flake8/ruff/black) and basic type hints.
   - Optional: lightweight unit tests if present.

2. **CI: Build & Package**
   - Build Docker image for FastAPI + model serving.
   - Tag images by branch, commit SHA, and `latest` for main.

3. **Model Evaluation (Optional per change or scheduled)**
   - Run training/eval on `data/master_airquality_clean.csv` if available or provided via remote.
   - Log metrics/artifacts (metrics JSON, predictions, SHAP plots) to `artifacts/` and experiment tracker.
   - Enforce acceptance gates: RMSE/MAE/R²/MAPE targets, fairness thresholds (DI/SPD), latency budget.

4. **Security & Policy Checks**
   - Scan dependencies (pip audit/OSS index).
   - Validate no secrets committed (secret scanning).

5. **Deploy: Staging**
   - Pull image from registry (e.g., GHCR).
   - Apply environment config/secrets; run DB/data connectivity smoke.
   - Health + `/metrics` checks; basic prediction smoke test.

6. **Deploy: Production**
   - Promote image tag that passed staging.
   - Rolling restart or blue/green; verify health/latency/error-rate SLOs.

7. **Post-Deploy Verification**
   - Monitor Prometheus/Grafana dashboards for latency, error rate, throughput.
   - Optional canary window with automatic rollback on SLO breach.

## Environments
- **CI runners**: ephemeral; require access to artifact storage and (optional) DVC remote for data.
- **Staging**: mirrors production configs with safe secrets and limited scale.
- **Production**: scaled deployment with monitoring/alerting; controlled secrets.

## Triggers
- **On push/PR**: CI lint/test/build; optional eval if data available.
- **Manual dispatch**: training/eval workflows with data input; deploy workflow with tag selection.
- **Scheduled (cron)**: drift/eval jobs to refresh metrics and prompt retraining.

## Automating Model Training and Testing
- Split fast vs full runs: PRs run quick checks (subset data or cached model); scheduled/manual runs do full training/eval.
- Data access: fetch via DVC/remote URL when enabled; otherwise reuse committed sample data for smoke.
- Reproducibility: fix seeds, log data hash, code SHA, hyperparams, and env (Python/package versions).
- Compute profile: prefer CPU for quick eval; allow GPU runners for full jobs when available.
- Artifacts: write metrics/predictions/plots to `artifacts/`; push best model to `models/` with metadata for traceability.
- Gates: fail the job if RMSE/MAE/R²/MAPE or fairness thresholds regress beyond allowed deltas; publish comparison to last approved run.
- Notifications: surface failures in CI and send alerts to the owning channel; include links to artifacts and dashboards.

## Building and Running a Deployment Pipeline
- Prereqs: registry access (GHCR or similar), Dockerfile, env config for staging/prod, SSH or K8s access, secrets injected via CI.
- Build: `docker build` with tags for `branch`, `commit`, and (for main) `latest`; include model artifact/version metadata in labels.
- Push: authenticate to registry, push all tags; mark which tag is promotion candidate (post-staging).
- Deploy to staging: pull candidate tag; apply env vars/secrets; run migrations if any; `docker compose up -d` or `kubectl apply`.
- Health checks: wait for `/health` and `/metrics`; run a prediction smoke test using a canned payload; fail fast on errors.
- Promote to prod: reuse the exact image tag that passed staging; rollout restart/blue-green; keep N−1 tag ready.
- Post-deploy: verify SLOs (latency/error rate) and model ID in logs/metrics; notify channel with image tag, model version, and links.

## Continuous Integration Hands-On Exercise (project-specific)
1. Create a branch `ci-exercise` and add a minimal workflow file at `.github/workflows/ci.yml`.
2. Install deps locally with `python -m pip install -r requirements.txt` to mirror runner setup.
3. Add lint step using `ruff` or `flake8` over `scripts/`, `kubeflow_pipeline.py`, `kubeflow_deploy.py`, and `main.py` to catch style/errors early.
4. Add a unit/test step that imports the pipeline entry points: `python -m pytest -q scripts tests` (if tests dir absent, create a smoke test that imports `kubeflow_pipeline.py` and `validate_pipeline.py`).
5. Add a data schema smoke: load a small sample from `data/master_airquality_clean.csv` (first 200 rows) and assert expected columns before training.
6. Build the Docker image from `Dockerfile` tagged with `ci-${{ github.sha }}`; ensure it copies `models/` and `artifacts/` placeholders.
7. Run a containerized smoke test: start the image, call `/health` and a sample prediction using a payload from `artifacts/test_predictions.csv` (one row) or a hardcoded JSON.
8. Cache pip packages and Docker layers to speed subsequent runs; use Actions cache keys scoped to `requirements.txt` hash.
9. Upload artifacts: lint/test reports, `artifacts/evaluation_metrics.json` if generated, and container logs for debugging.
10. Add branch protection so PRs require the CI workflow to pass before merge.
11. Optional stretch: add a scheduled workflow that runs `validate_pipeline.py` nightly, publishes metrics to `artifacts/`, and opens an issue if regression is detected.

## Inputs & Secrets
- Registry credentials (GHCR PAT) for push/pull.
- SSH credentials if using remote deploy via Compose/SSH.
- Experiment tracking API keys (Comet/W&B) for logging runs.
- Optional data URL or credentials if data is pulled during CI.

## Artifacts & Outputs
- Docker image tagged by branch, commit, and `latest` for main.
- Metrics and plots in `artifacts/` (evaluation_metrics.json, predictions CSVs, SHAP PNGs).
- Model binaries in `models/` (e.g., best_pm25_model.pkl) plus metadata.
- CI logs and reports (lint, tests, security scans).

## Rollback Strategy
- Keep previous image tag and model artifact (N−1) available.
- If SLO breach or regression: redeploy N−1 tag; document rollback in changelog.
- Maintain run/commit references for the promoted and rollback versions.

## Local Verification (Before Commit)
```powershell
# Lint / format (example if configured)
python -m pip install -r requirements.txt
flake8 .

# Build image locally
docker build -t pm25:local .

# Smoke test API
docker run --rm -p 8000:8000 pm25:local
Invoke-RestMethod -Method Get -Uri http://localhost:8000/
```

## Notes for GitHub Actions (if used)
- CI workflow typically: setup Python, install deps, lint, optional tests, build image, (optional) eval.
- Docker build/push workflow: build from `Dockerfile`, push to GHCR `ghcr.io/<owner>/mlops-project:{tag}`.
- Deploy workflow (Compose/SSH): pull image, `docker compose up -d` on target host; needs SSH secrets and GHCR PAT.

## Tools Setup: GitHub Actions / Jenkins / GitLab CI

### GitHub Actions
- Workflows: YAML under `.github/workflows/` (CI, docker-build, deploy, eval).
- Runners: `ubuntu-latest` with Python setup and Docker available.
- Secrets: `GHCR_PAT`, `SSH_HOST/USER/PRIVATE_KEY`, experiment tracking keys, optional data URL.
- Caching: pip cache and Docker layers (setup-buildx + cache-from/to) to speed builds.
- Triggers: `push`/`pull_request` for CI; `workflow_dispatch` for training/eval/deploy; `schedule` for drift/eval.

### Jenkins
- Agents: Docker-capable nodes with Python 3.8+ and git.
- Pipeline: Declarative Jenkinsfile stages mirroring Actions (lint/test, build/push image, optional eval, deploy via SSH/Compose).
- Credentials: store GHCR PAT/SSH keys/API keys in Jenkins Credentials; inject via withCredentials.
- Caching: Reuse workspace for pip cache; Docker layer caching via shared registry.
- Triggers: SCM webhooks or cron; promote jobs chained after staging verification.

### GitLab CI
- Runners: Docker-in-Docker or Docker socket for builds; Python image for lint/eval.
- Config: `.gitlab-ci.yml` stages (lint → build → eval → deploy).
- Variables: store registry creds, SSH keys, tracking API keys in GitLab CI/CD variables (masked/protected).
- Caching: `cache:` for pip; `services: docker:dind` for image builds; use `--cache-from` for layer reuse.
- Triggers: `only/except` or `rules`; scheduled pipelines for drift/eval; manual jobs for deploy/promote.

### Common Patterns
- Separate CI (lint/test) from build/push and from deploy; gate deploy on staging checks.
- Always tag images with commit SHA and branch; promote by tag, not by rebuilding.
- Keep data access optional in CI; allow supplying data URL/DVC pull when needed for eval.
- Capture metrics/artifacts in a known path (`artifacts/`) and upload as job artifacts.

## Acceptance Gates (Promote/Deploy)
- Meets metric targets (RMSE/MAE/R²/MAPE) on holdout.
- Fairness within target bands (DI 0.8–1.25; SPD < 0.1).
- Latency P95 < 100ms; error rate < 1%; uptime SLO met in staging burn-in.
- Governance artifacts updated: MODEL_CARD, AUDIT_CHECKLIST, changelog/registry entry.

## Traceability
- Tag images with commit SHA; record data version (DVC hash/date) and run ID in release notes.
- Keep `models/model_metadata.json` aligned with the served model in the image.
- Log model version in app metrics/logs for correlation with incidents.
