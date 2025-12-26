# Model Retraining & Redeployment Plan

## 1. Retraining Strategy

### 1.1 Triggers
Retraining is triggered by one of the following events:
- **Scheduled**: Monthly retraining to incorporate new data.
- **Performance Drift**: If model performance (RMSE) degrades by >10% compared to the baseline on the monitoring dashboard.
- **Data Drift**: If the drift score (calculated via Evidently AI) for key features (`PM10`, `O3`) exceeds the threshold (0.1).

### 1.2 Data Window
- **Rolling Window**: Train on the last 12 months of data to capture seasonal trends while remaining current.
- **Validation**: Use the most recent 1 month for validation.

## 2. Evaluation Gates

Before a new model version is promoted to production, it must pass the following gates:
1.  **Metric Improvement**: The new model must show equal or better RMSE/MAE than the currently deployed model on the holdout test set.
2.  **Fairness Check**: The Disparate Impact ratio must remain within the range [0.8, 1.25].
3.  **Latency Check**: Inference time must not exceed 100ms (P95).

## 3. Redeployment Process

### 3.1 Continuous Deployment (CD) Pipeline
1.  **Build**: A new Docker image is built with the updated model artifact.
2.  **Test**: Automated unit and integration tests run.
3.  **Staging**: Deploy to a staging environment.
4.  **Approval**: Manual or automated sign-off based on evaluation gates.
5.  **Production**: Rolling update to production (Kubernetes/Docker Compose).

### 3.2 Rollback
If the error rate spikes >5% post-deployment:
1.  Revert to the previous Docker image tag.
2.  Mark the new model version as "Rejected" in the model registry.

## 4. Feedback Loop
- **Ground Truth Collection**: Actual PM2.5 values are collected from reference stations daily.
- **Residual Analysis**: Residuals (Predicted - Actual) are analyzed to identify systematic errors (e.g., underprediction during holidays).
