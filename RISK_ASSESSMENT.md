# Risk Assessment: Air Quality Prediction System

## 1. Methodology
We utilize a standard **5x5 Risk Matrix** to evaluate risks based on two dimensions:
- **Likelihood**: Rare (1) to Almost Certain (5)
- **Impact**: Insignificant (1) to Catastrophic (5)

**Risk Score = Likelihood × Impact**
- **Low (1-4)**: Acceptable risk.
- **Medium (5-9)**: Monitor and mitigate.
- **High (10-19)**: Urgent action required.
- **Extreme (20-25)**: Stop deployment until resolved.

## 2. Top Risks Identified

### 2.1 Data Drift (Score: 16 - High)
- **Description**: The distribution of input features (e.g., PM10, O3) changes over time due to seasonal shifts or new pollution sources.
- **Likelihood**: 4 (Likely)
- **Impact**: 4 (Major - Model accuracy degrades significantly)
- **Mitigation**:
    - Implement automated drift detection using Evidently AI.
    - Schedule monthly retraining.
    - Alert when drift score > 0.1.

### 2.2 Model Bias (Score: 12 - High)
- **Description**: The model performs significantly worse for specific time periods (e.g., weekends) or locations.
- **Likelihood**: 3 (Possible)
- **Impact**: 4 (Major - Reputational damage and poor decision making)
- **Mitigation**:
    - Conduct fairness audit using AIF360 (Disparate Impact analysis).
    - Include fairness metrics in the model card.
    - Reject models with Disparate Impact < 0.8 during evaluation.

### 2.3 API Latency Spikes (Score: 9 - Medium)
- **Description**: Inference time exceeds 500ms during high load.
- **Likelihood**: 3 (Possible)
- **Impact**: 3 (Moderate - Poor user experience)
- **Mitigation**:
    - Horizontal scaling of FastAPI containers (Kubernetes/Docker Swarm).
    - Cache frequent predictions.
    - Monitor P95 latency via Prometheus.

### 2.4 Sensor Failure / Missing Data (Score: 6 - Medium)
- **Description**: Upstream sensors fail, leading to missing values in input.
- **Likelihood**: 3 (Possible)
- **Impact**: 2 (Minor - Model can handle some missingness)
- **Mitigation**:
    - Robust imputation strategy (median filling) in the inference pipeline.
    - Data quality checks before inference.

## 3. Risk Matrix Visualization
See `risk_matrix_5x5.html` for an interactive visualization of these risks.
