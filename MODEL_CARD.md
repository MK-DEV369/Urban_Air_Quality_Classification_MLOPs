# Model Card: PM2.5 Prediction Model

## Model Details
- **Model name**: best_pm25_model.pkl
- **Model type**: (auto-filled at runtime) e.g., XGBRegressor
- **Version**: v1
- **Framework**: scikit-learn / xgboost
- **Location**: `models/best_pm25_model.pkl`

## Intended Use
- Predict PM2.5 concentration using measured pollutants + time features.

## Inputs
Expected features used by the API:
- `PM10` (float)
- `O3` (float)
- `CO` (float)
- `hour` (int)
- `dayofweek` (int)
- `month` (int)

## Output
- `PM25_prediction` (float)

## Training Data
- Reference dataset: `data/master_airquality_clean.csv`
- Preprocessing: performed in `clean.ipynb` (missing values handling + time features)

## Evaluation
- Metrics reported in: W&B runs and training notebook (`train1.ipynb`)

## Ethical Considerations
- Bias/fairness analysis: `governance.ipynb`
- Governance report: `governance_report.json`

## Limitations
- Model quality depends on sensor coverage and data quality.
- Temporal/seasonal drift may degrade performance.

## Monitoring
- Operational metrics: `/metrics` (Prometheus)
- Drift checks: `monitoring/evidently_drift_report.py` (optional)
