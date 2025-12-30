
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
# Building and Evaluating an ML Model

## 1. Model Development Workflow

```
Data → Feature Engineering → Model Training → Evaluation → Deployment
  ↓                              ↓                ↓            ↓
Clean  →  Extract Features  →  Train/Val/Test → Metrics → Registry
```

## 2. Model Selection

### Candidate Models

We evaluated three regression models for PM2.5 prediction:

| Model | Type | Pros | Cons |
|-------|------|------|------|
| **Linear Regression** | Linear | Fast, interpretable | Limited expressiveness |
| **Random Forest** | Ensemble (Bagging) | Robust, handles non-linearity | Slower inference |

## Appendix: Training, Validation, and Testing (Detailed Guide)

### A. Dataset & Prerequisites
- Cleaned dataset path: `data/master_airquality_clean.csv`
- Features: `PM10`, `O3`, `CO`, `hour`, `dayofweek`, `month`
- Target: `PM2.5`
- Environment: Python 3.8+, packages from `requirements.txt`

### B. Split Strategy
- Use a temporal split to avoid leakage and mimic real‑world scenarios.

```python
# Temporal split (no shuffling)
| **XGBoost** | Ensemble (Boosting) | High performance, regularization | Black-box, tuning needed |

### Selection Criteria
1. **Predictive Performance**: RMSE, R²
2. **Training Time**: < 30 minutes on CPU
3. **Inference Speed**: < 100ms per prediction
4. **Interpretability**: Can we explain predictions?

---

Optional time‑series cross‑validation:

```python

## 3. Training Process

### Data Splits

```python
# Temporal split (no shuffling)
n = len(df)
train_size = int(0.6 * n)  # 60%

### C. Training
- Train candidate models (Linear, Random Forest, XGBoost) on `train_df`.
- Track hyperparameters, metrics, and artifacts via W&B/Comet.

Windows PowerShell examples:

```powershell
val_size = int(0.2 * n)    # 20%
test_size = n - train_size - val_size  # 20%
```

**Why temporal split?**
- Prevents data leakage from future to past
- Mimics real-world deployment (predict future from past)
- More realistic performance estimate


### D. Validation
- Evaluate on `val_df` to tune hyperparameters and avoid overfitting.
- Metrics: RMSE, MAE, R², MAPE.
- Visuals: predicted vs actual, residual histogram, residuals vs predicted, error by hour.
- Interpretability: SHAP summary/dependence plots.

```powershell
### Training Configuration


### E. Testing (Holdout)
- Final performance on `test_df` only after model selection.
- Log metrics and confusion analyses (for regression: error distributions).
- Store outputs: `artifacts/evaluation_metrics.json`, `artifacts/test_predictions.csv`, `artifacts/worst_predictions.csv`.

### F. Acceptance Criteria
- Meets targets in Evaluation Metrics: RMSE < 30 µg/m³, MAE < 20 µg/m³, R² > 0.75, MAPE < 40%.
- Operational constraints: P95 latency < 100ms; error rate < 1%.
- Fairness targets: Disparate Impact within 0.8–1.25; Statistical Parity Difference < 0.1.

### G. Reproducibility & Versioning
- Pin dependency versions; set random seeds.
- Record data version (DVC or date‑tag) and code hash.
- Save `models/best_pm25_model.pkl` and associated metadata JSON.

### H. Troubleshooting Tips
- Large CSVs: use chunked reads or memory‑efficient dtypes.
- Timestamp parsing: coerce errors and validate timezone consistency.
- Missing values: confirm imputation for selected features and target handling.
- Outliers: verify percentile bounds do not remove legitimate high pollution events.
#### Linear Regression
```python
LinearRegression()
# No hyperparameters to tune
```

#### Random Forest
```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Tree depth
    min_samples_split=2,   # Min samples to split
    random_state=42
)
```

#### XGBoost (Best Model)
```python
XGBRegressor(
    n_estimators=300,        # Boosting rounds
    learning_rate=0.05,      # Shrinkage
    max_depth=7,             # Tree depth
    subsample=0.9,           # Row sampling
    colsample_bytree=0.9,    # Column sampling
    tree_method='hist',      # Fast algorithm
    objective='reg:squarederror'
)
```

### Training Time
- **Linear Regression**: ~30 seconds
- **Random Forest**: ~10 minutes
- **XGBoost**: ~15 minutes

---

## 4. Model Evaluation

### Evaluation Metrics

#### Regression Metrics
```
RMSE (Root Mean Squared Error) = sqrt(mean((y_pred - y_true)²))
MAE (Mean Absolute Error)       = mean(|y_pred - y_true|)
R² (Coefficient of Determination) = 1 - SS_res / SS_tot
MAPE (Mean Absolute Percentage Error) = mean(|y_pred - y_true| / y_true) * 100
```

### Results Comparison

| Model | RMSE (↓) | MAE (↓) | R² (↑) | MAPE (↓) |
|-------|----------|---------|--------|----------|
| Linear Regression | 68.23 | 31.45 | 0.24 | 72.5% |
| Random Forest | 54.87 | 24.12 | 0.48 | 58.3% |
| **XGBoost** | **52.15** | **23.24** | **0.53** | **55.3%** |

**Winner**: XGBoost (lowest RMSE)

### Visualization

Generated plots (see `artifacts/`):

1. **Predicted vs Actual** (`pred_vs_actual.png`)
   - Scatter plot showing correlation
   - Ideal: points on diagonal line
   - Current: Good fit with some scatter

2. **Residual Distribution** (`residuals.png`)
   - Histogram of errors
   - Ideal: Centered at zero, bell-shaped
   - Current: Slight positive skew

3. **Residuals vs Predicted** (`residuals.png`)
   - Check for heteroscedasticity
   - Ideal: Random scatter around zero
   - Current: Some patterns at high PM2.5

4. **Error by Hour** (`error_by_hour.png`)
   - MAE varies by time of day
   - Higher errors during rush hours (7-9 AM, 6-8 PM)

---

## 5. Error Analysis

### Worst Predictions

Analyzed top 10 worst predictions:

| Actual PM2.5 | Predicted PM2.5 | Error | Time | Likely Cause |
|--------------|-----------------|-------|------|--------------|
| 342.1 | 180.5 | 161.6 | 8 AM | Traffic surge event |
| 298.7 | 165.2 | 133.5 | 7 PM | Diwali fireworks |
| 276.3 | 158.9 | 117.4 | 6 AM | Construction dust |

**Common failure modes**:
1. **Extreme events**: Festivals, accidents, construction
2. **Missing context**: Weather changes, policy interventions
3. **Sensor errors**: Outliers in input data

### Error Patterns by Hour

```
Hour  | Mean Error | Interpretation
------|------------|-----------------------------------
0-5   | 18.5 µg/m³ | Low error (low activity)
6-9   | 28.3 µg/m³ | High error (rush hour variability)
10-16 | 20.1 µg/m³ | Moderate error (steady state)
17-20 | 26.7 µg/m³ | High error (evening rush)
21-23 | 19.4 µg/m³ | Low error (decreasing activity)
```

---

## 6. Experiment Tracking with Comet ML

### What is Logged?

1. **Hyperparameters**
   ```python
   experiment.log_parameters({
       'n_estimators': 300,
       'learning_rate': 0.05,
       'max_depth': 7
   })
   ```

2. **Metrics**
   ```python
   experiment.log_metrics({
       'rmse': 52.15,
       'mae': 23.24,
       'r2': 0.53
   })
   ```

3. **Model Artifacts**
   ```python
   experiment.log_model("XGBoost", "models/xgb_reg.joblib")
   ```

4. **Code Version**
   - Automatically tracks Git commit hash
   - Logs changed files

### Accessing Results

1. Go to [comet.ml](https://www.comet.ml)
2. Navigate to project: `pm25-airquality`
3. View experiment runs, compare metrics, visualize learning curves

---

## 7. Model Interpretability (SHAP)

### Feature Importance

```
Feature   | SHAP Importance | Interpretation
----------|-----------------|-------------------------------
PM10      | 0.65            | Primary predictor (coarse PM)
CO        | 0.15            | Industrial activity marker
O3        | 0.10            | Photochemical reactions
hour      | 0.06            | Diurnal patterns
month     | 0.03            | Seasonal effects
dayofweek | 0.01            | Minimal impact
```

### SHAP Visualizations

1. **Summary Plot** (`artifacts/shap_summary.png`)
   - Shows feature contributions for all predictions
   - Color: Feature value (red=high, blue=low)
   - X-axis: SHAP value (impact on prediction)

2. **Dependence Plots** (`artifacts/shap_dependence_*.png`)
   - PM10: Strong positive correlation
   - O3: Slight negative correlation (scavenges PM2.5)
   - Hour: U-shaped (peaks at rush hours)

### Interpreting Individual Predictions

Example: Prediction for a specific observation

```python
Base value (average): 60.2 µg/m³
+ PM10 contribution:   +35.8
+ CO contribution:     +8.3
+ O3 contribution:     -2.1
+ hour contribution:   +3.5
+ month contribution:  +1.2
+ dayofweek contrib.:  +0.3
= Final prediction:    107.2 µg/m³
```

---

## 8. Model Validation

### Cross-Validation (Optional)

While we used a single temporal split, k-fold CV can provide more robust estimates:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)

print(f"CV R²: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### Overfitting Check

```
Metric     | Train | Validation | Test | Interpretation
-----------|-------|------------|------|----------------
RMSE       | 45.2  | 51.3       | 52.1 | Slight overfitting
R²         | 0.62  | 0.54       | 0.53 | Generalizes well
```

**Conclusion**: Model is not severely overfitting.

---

## 9. Model Selection Decision

### Why XGBoost?

1. **Best Performance**: Lowest RMSE (52.15) and highest R² (0.53)
2. **Reasonable Training Time**: 15 minutes is acceptable
3. **Inference Speed**: 50ms per prediction (meets <100ms requirement)
4. **Interpretability**: SHAP provides good explanations
5. **Robustness**: Built-in regularization prevents overfitting

### Trade-offs

| Aspect | Linear Regression | Random Forest | XGBoost |
|--------|-------------------|---------------|---------|
| Performance | ❌ Poor | ✅ Good | ✅✅ Best |
| Speed | ✅✅ Very Fast | ⚠️ Moderate | ✅ Fast |
| Interpretability | ✅✅ High | ✅ Moderate | ✅ Moderate (with SHAP) |
| Tuning Effort | ✅✅ None | ⚠️ Some | ⚠️ Significant |

---

## 10. Model Saving and Versioning

### Saving the Model

```python
import joblib

# Save model
joblib.dump(model, "models/best_pm25_model.pkl")

# Save metadata
metadata = {
    "model_type": "XGBoost",
    "features": ["PM10", "O3", "CO", "hour", "dayofweek", "month"],
    "metrics": {"rmse": 52.15, "mae": 23.24, "r2": 0.53},
    "trained_on": "2024-12-24",
    "data_version": "v1"
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f)
```

### Loading the Model

```python
model = joblib.load("models/best_pm25_model.pkl")
prediction = model.predict([[120, 50, 2.5, 8, 0, 12]])  # Example input
```

### Version Control

Registered in custom model registry:
```bash
python scripts/model_registry.py
```

See `models/registry/` for versioned models.

---

## 11. Next Steps

1. **Hyperparameter Tuning**: Run `scripts/hyperparameter_tuning.py`
2. **Deploy API**: `docker-compose up`
3. **Monitor Performance**: Check Prometheus/Grafana
4. **Retrain Periodically**: Use `RETRAINING_PLAN.md`

---

## References

---

## Appendix: Evaluation Metrics & Visualization (Detailed Guide)

### A. Metrics Overview
- **RMSE** (Root Mean Squared Error): $\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$
- **MAE** (Mean Absolute Error): $\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i|$
- **R²** (Coefficient of Determination): $R^2 = 1 - \frac{\sum(\hat{y}-y)^2}{\sum(y-\bar{y})^2}$
- **MAPE** (Mean Absolute Percentage Error): $\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{\hat{y}_i - y_i}{y_i}\right|$

Targets (holdout test): RMSE < 30 µg/m³, MAE < 20 µg/m³, R² > 0.75, MAPE < 40%.

Segment analyses (recommended): compute metrics per `hour`, `dayofweek`, and `month` to assess temporal fairness and robustness.

### B. Generation Commands
Evaluate trained models and emit artifacts:

```powershell
python scripts\evaluate_model.py --data data\master_airquality_clean.csv --model models\best_pm25_model.pkl --outdir artifacts
```

Run SHAP interpretability:

```powershell
python scripts\shap_analysis.py --model models\best_pm25_model.pkl --data data\master_airquality_clean.csv --outdir artifacts
```

Outputs:
- Metrics JSON: `artifacts/evaluation_metrics.json`
- Predictions CSV: `artifacts/test_predictions.csv`
- Worst errors CSV: `artifacts/worst_predictions.csv`
- Plots: saved under `artifacts/`

### C. Visualization Catalog
- **Predicted vs Actual** (`artifacts/pred_vs_actual.png`)
   - Purpose: overall fit; ideal points lie on the diagonal.
   - Red flags: systematic bias (consistent offset) or strong curvature.

- **Residual Distribution** (`artifacts/residuals.png`)
   - Purpose: error spread; ideal centered at zero with bell shape.
   - Red flags: heavy tails or skew indicating outliers or bias.

- **Residuals vs Predicted** (plot in `artifacts/`)
   - Purpose: heteroscedasticity check; ideal random scatter around zero.
   - Red flags: funnel shapes or bands at high PM2.5 values.

- **Error by Hour** (`artifacts/error_by_hour.png`)
   - Purpose: diurnal reliability; rush-hour spikes often occur.
   - Action: consider feature augmentation if systematic peaks persist.

- **SHAP Summary** (`artifacts/shap_summary.png`)
   - Purpose: global feature importance and directionality.
   - Typical: `PM10` dominant; `CO`, `O3` secondary; temporal features show periodic effects.

- **SHAP Dependence** (`artifacts/shap_dependence_*.png`)
   - Purpose: non-linear relationships per feature.
   - Watch: unexpected sign flips or sharp discontinuities (data quality issues).

### D. Interpretation Guidance
- Confirm that improvements in RMSE also reduce MAE; large RMSE with modest MAE indicates few extreme errors.
- Use R² for comparative fit but prioritize RMSE/MAE for decision thresholds.
- Investigate segments where errors spike (hours, months) and consider additional features or reweighting.
- When SHAP highlights unexpected drivers, revisit preprocessing and missingness.

### E. Operational Visualization
- **Prometheus/Grafana**: monitor API latency (P95), error rate, and throughput; align alerts with thresholds.
- Periodically overlay ground truth vs predictions (batch comparison) to assess drift; trigger retraining per plan.

### F. Automation & Reproducibility
- Log plots and metrics with experiment tracking (W&B/Comet).
- Version the dataset (DVC or date tags) referenced in artifact metadata.
- Store model metadata (features, metrics, training date) next to plots for auditability.

---

## Appendix: Experiment Tracking (Detailed Guide)

### A. Purpose
- Ensure reproducibility and auditability across runs.
- Tie code, data version, hyperparameters, metrics, and artifacts to each experiment.

### B. Tools
- Primary: Comet ML (scripted flow via `scripts/train_with_comet.py`).
- Optional: Weights & Biases (configure `wandb login` similarly if preferred).

### C. What to Log
- **Hyperparameters**: model type, estimators/depth, learning rate, subsample, colsample, seeds.
- **Metrics**: RMSE, MAE, R², MAPE (train/val/test if available).
- **Artifacts**: model file (`models/best_pm25_model.pkl`), metrics JSON (`artifacts/evaluation_metrics.json`), predictions (`artifacts/test_predictions.csv`, `artifacts/worst_predictions.csv`), plots (SHAP, residuals).
- **Data version**: DVC hash or date tag for `data/master_airquality_clean.csv`.
- **Code version**: Git commit hash; optionally diff summary.
- **Environment**: Python version, key package versions (xgboost, sklearn, pandas, numpy).

### D. Running Tracked Training (PowerShell)

```powershell
# Set Comet API key (one-time per shell)
$env:COMET_API_KEY = "<your_api_key>"

# Train with tracking
python scripts\train_with_comet.py --data data\master_airquality_clean.csv --outdir artifacts

# Optional: Evaluate and log results as an additional experiment step
python scripts\evaluate_model.py --data data\master_airquality_clean.csv --model models\best_pm25_model.pkl --outdir artifacts
```

If using W&B instead:

```powershell
wandb login
python scripts\train_with_comet.py --data data\master_airquality_clean.csv --outdir artifacts  # swap to your W&B-enabled script if present
```

### E. Run Hygiene
- Use a clear naming convention: `{model}_{date}_{commit}` (e.g., `xgb_2024-12-24_ab12cd`).
- Tag runs with data version (`data_vYYYYMMDD` or DVC hash), environment (`py3.10`), and purpose (`baseline`, `tuning`, `ablation`).
- Upload artifacts after evaluation; keep metrics and plots in the same run for traceability.

### F. Promotion Criteria
- Promote a model only if it meets acceptance criteria (RMSE/MAE/R²/MAPE thresholds, latency, fairness targets) and artifacts are logged.
- Record the promoted model’s run ID and copy to `models/best_pm25_model.pkl` with metadata.

### G. Troubleshooting
- **Authentication failures**: ensure `COMET_API_KEY`/`WANDB_API_KEY` is set in the shell/CI secrets.
- **Network issues**: retry with stable connectivity; consider offline logging and later sync.
- **Large artifacts**: prune or compress plots/CSVs before upload; adjust artifact limits if needed.
- **Mismatched versions**: log package versions; if metrics shift unexpectedly, check data hash and commit.

### H. CI Considerations
- In CI, inject API keys as secrets and set them as env vars before training/eval steps.
- Allow runs to upload only metrics/artifacts (no PII); ensure data remotes are accessible for `dvc pull`.

---

## Appendix: Interpreting Model Results & Error Analysis

### A. Read the Core Metrics
- Check holdout metrics first: RMSE, MAE, R², MAPE vs targets (RMSE < 30, MAE < 20, R² > 0.75, MAPE < 40%).
- Compare train/val/test to spot over/under-fitting; small gaps suggest good generalization.

### B. Inspect Residuals and Distributions
- **Residual histogram**: should center near 0 with light tails; heavy tails signal outliers or missed features.
- **Residuals vs Predicted**: look for funnel shapes (heteroscedasticity) or curvature (unmodeled non-linearity).
- **Predicted vs Actual**: diagonal alignment indicates calibration; systemic offsets imply bias.

### C. Segment Error Analysis
- Slice errors by `hour`, `dayofweek`, `month` (see `artifacts/error_by_hour.png`).
- Identify segments with elevated MAE; consider feature augmentation or segment-aware thresholds.
- If available, slice by location/station to detect spatial weaknesses.

### D. Worst-Case Review
- Use `artifacts/worst_predictions.csv` to examine top errors.
- Label likely causes (festivals, weather shifts, sensor glitches); decide on data cleaning or feature additions.

### E. Interpretability (SHAP)
- **Summary plot** (`artifacts/shap_summary.png`): confirms global drivers; `PM10` should dominate; `CO`, `O3` secondary; temporal features show periodicity.
- **Dependence plots** (`artifacts/shap_dependence_*.png`): check for sensible monotonic/curved relationships; unexpected sign flips may mean data issues.

### F. Drift and Stability Checks
- Compare recent batch predictions vs historical error distributions; large shifts may indicate drift.
- Monitor key feature distributions (PM10/CO/O3) and temporal segments for shifts; trigger retraining per plan.

### G. Action Playbook
- **High rush-hour errors**: add congestion/weather proxies; consider per-segment calibration.
- **Outlier-driven RMSE**: tighten outlier handling or add robust loss; inspect sensors.
- **Bias across temporal segments**: revisit fairness targets; adjust thresholds or retrain with reweighting.
- **Latency spikes**: optimize model or scale replicas if error analysis shows acceptable quality but poor QoS.

### H. Reporting & Traceability
- Capture metrics, plots, and SHAP outputs in the experiment run (Comet/W&B) with data/code hashes.
- Note any manual exclusions (e.g., dropped extreme events) in the run notes and governance artifacts.

### I. Quick Commands

```powershell
# Evaluate and produce residuals/plots
python scripts\evaluate_model.py --data data\master_airquality_clean.csv --model models\best_pm25_model.pkl --outdir artifacts

# Generate SHAP visualizations
python scripts\shap_analysis.py --model models\best_pm25_model.pkl --data data\master_airquality_clean.csv --outdir artifacts
```

---

## Appendix: Hyperparameter Tuning & Performance

### A. Goals
- Improve RMSE/MAE/R² while respecting latency and stability.
- Avoid overfitting; prefer compact models for API performance.

### B. Suggested Search Spaces
- **Random Forest**: `n_estimators` [100, 300], `max_depth` [10, 20, None], `min_samples_split` [2, 5], `max_features` ["auto", "sqrt"].
- **XGBoost**: `n_estimators` [200, 400], `learning_rate` [0.03, 0.05, 0.1], `max_depth` [5, 7, 9], `subsample` [0.7, 0.9, 1.0], `colsample_bytree` [0.7, 0.9, 1.0], `min_child_weight` [1, 5], `reg_alpha` [0, 0.1], `reg_lambda` [1, 2].

### C. Commands (PowerShell)

```powershell
python scripts\hyperparameter_tuning.py --data data\master_airquality_clean.csv --outdir artifacts
```

### D. Evaluation During Tuning
- Use temporal validation; log val RMSE/MAE/R²/MAPE per trial.
- Enforce early stopping for XGBoost when val metric stalls.
- Track best trial parameters and seed for reproducibility.

### E. Performance Considerations
- Depth/estimators heavily impact latency; benchmark P95 prediction time after selecting a candidate.
- Prefer `tree_method='hist'` for XGBoost on CPUs; consider pruning estimators if latency is high.
- Cache model load and precompute request-derived features (hour/dayofweek/month) in the API.

### F. Logging & Artifacts
- Log each trial’s hyperparameters, metrics, and artifacts to the experiment tracker.
- Save the best model and metadata: params, data version, code hash, val/test metrics.

### G. Promotion Checklist Post-Tuning
- New model beats current on holdout metrics without violating latency/fairness targets.
- Artifacts updated: metrics JSON, predictions CSVs, SHAP plots, run notes.
- Governance updated if promoted: MODEL_CARD, AUDIT_CHECKLIST, registry entry.

---

## Appendix: Interpretability, Model Versioning, and Registry

### A. Interpretability (SHAP)
- Purpose: explain global and local drivers; validate that learned patterns align with domain expectations.
- Commands:

```powershell
python scripts\shap_analysis.py --model models\best_pm25_model.pkl --data data\master_airquality_clean.csv --outdir artifacts
```

- Key artifacts: `artifacts/shap_summary.png`, `artifacts/shap_dependence_*.png`.
- Reading: `PM10` should dominate; `CO`, `O3` secondary; temporal features show periodic effects. Unexpected sign flips or spikes suggest data quality issues.
- Local explanations: use SHAP force plots or decision plots (extend script if needed) to inspect specific worst cases from `artifacts/worst_predictions.csv`.

### B. Model Versioning
- Store models under `models/` with clear naming: `best_pm25_model.pkl`, `xgb_reg.json`, `rf_reg.joblib`.
- Record metadata alongside the model (JSON):
   - model type, hyperparameters
   - features used
   - metrics (train/val/test)
   - data version (DVC hash or date tag)
   - code commit hash
   - training date and seed
- Example save snippet:

```python
import joblib, json, pathlib

joblib.dump(model, "models/best_pm25_model.pkl")
meta = {
      "model_type": "XGBoost",
      "features": ["PM10", "O3", "CO", "hour", "dayofweek", "month"],
      "metrics": {"rmse": 52.15, "mae": 23.24, "r2": 0.53, "mape": 55.3},
      "data_version": "data_v2024_12_24",
      "code_commit": "<git-hash>",
      "trained_on": "2024-12-24",
      "seed": 42
}
pathlib.Path("models/model_metadata.json").write_text(json.dumps(meta, indent=2))
```

### C. Registry Practices
- Lightweight registry: use `models/registry/` or versioned filenames plus a manifest file.
- Promotion flow:
   - Evaluate candidate; ensure it passes acceptance (metrics, latency, fairness).
   - Copy to `models/best_pm25_model.pkl` and update metadata/manifest with run ID, data version, code hash.
   - Archive previous model for rollback.
- Optional script: [scripts/model_registry.py](scripts/model_registry.py) to list/promote models.
- Traceability: link registry entries to experiment tracking run IDs and DVC data versions.

### D. Operational Integration
- API loading: [main.py](main.py) loads `models/best_pm25_model.pkl`; ensure the matching metadata accompanies deployments.
- Monitoring: log model version in Prometheus labels or logs to correlate metrics with specific releases.
- Rollback: keep N−1 model artifact ready; rollback by swapping the symlink/filename in `models/` and redeploying.

---

## Appendix: Documenting Model Updates

### A. Purpose
- Maintain traceability and auditability for every promoted model.
- Make it easy to compare versions and roll back if needed.

### B. Minimal Changelog Fields (per update)
- Model ID/name and file path (e.g., `models/best_pm25_model.pkl`).
- Run/experiment ID (Comet/W&B) and date.
- Data version (DVC hash or date tag) and code commit hash.
- Hyperparameters (diffs vs prior), training seed.
- Metrics (train/val/test: RMSE, MAE, R², MAPE) and fairness results (DI/SPD targets).
- Operational checks: P95 latency, uptime/error-rate status in staging.
- Artifacts paths: metrics JSON, predictions CSVs, SHAP plots, worst cases.
- Decision & rationale: why promoted; known limitations.
- Owner/approver and sign-off date.

### C. Where to Record
- Primary: experiment tracker run notes with links to artifacts.
- Repository: update `models/model_metadata.json` (or per-version metadata file) and maintain a simple changelog (e.g., `models/CHANGELOG.md` or manifest in `models/registry/`).
- Governance: update MODEL_CARD, AUDIT_CHECKLIST, and governance report for production promotions.

### D. Example Changelog Entry (YAML)

```yaml
- model: best_pm25_model.pkl
   run_id: comet:abc123
   data_version: dvc:3f2c9a1
   code_commit: a1b2c3d
   trained_on: 2024-12-24
   hyperparams:
      model_type: xgboost
      n_estimators: 300
      learning_rate: 0.05
      max_depth: 7
      subsample: 0.9
      colsample_bytree: 0.9
   metrics:
      rmse: 52.1
      mae: 23.2
      r2: 0.53
      mape: 55.3
   fairness:
      disparate_impact: 0.92
      statistical_parity_diff: 0.05
   ops:
      p95_latency_ms: 70
      error_rate: 0.3
   artifacts:
      metrics_json: artifacts/evaluation_metrics.json
      preds: artifacts/test_predictions.csv
      worst: artifacts/worst_predictions.csv
      shap: artifacts/shap_summary.png
   decision: "Promote: improved RMSE vs baseline; fairness within targets."
   owner: data.scientist@example.com
   approver: product.owner@example.com
   signoff_date: 2024-12-25
```

### E. Release Flow
1) Finish evaluation/interpretability; ensure targets met.
2) Update metadata and changelog entry; link run ID, data, code hash.
3) Copy/promote artifact to `models/best_pm25_model.pkl`; archive prior version for rollback.
4) Update governance artifacts (MODEL_CARD, AUDIT_CHECKLIST, governance report).
5) Deploy to staging; verify latency/error-rate; then promote to production.
6) Tag repo (optional) to mark the release commit.

### F. Rollback Notes
- Keep N−1 model and metadata; rollback by swapping the promoted file and redeploying.
- Note rollback decision, cause, and timestamp in the changelog.

- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- Scikit-learn Model Evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
