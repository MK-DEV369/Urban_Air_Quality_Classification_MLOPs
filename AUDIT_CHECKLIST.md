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
