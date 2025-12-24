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
| **XGBoost** | Ensemble (Boosting) | High performance, regularization | Black-box, tuning needed |

### Selection Criteria
1. **Predictive Performance**: RMSE, R²
2. **Training Time**: < 30 minutes on CPU
3. **Inference Speed**: < 100ms per prediction
4. **Interpretability**: Can we explain predictions?

---

## 3. Training Process

### Data Splits

```python
# Temporal split (no shuffling)
n = len(df)
train_size = int(0.6 * n)  # 60%
val_size = int(0.2 * n)    # 20%
test_size = n - train_size - val_size  # 20%
```

**Why temporal split?**
- Prevents data leakage from future to past
- Mimics real-world deployment (predict future from past)
- More realistic performance estimate

### Training Configuration

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

- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- Scikit-learn Model Evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
