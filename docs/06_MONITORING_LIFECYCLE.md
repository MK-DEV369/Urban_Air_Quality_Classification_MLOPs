# Monitoring and Model Lifecycle Management

## 1. Monitoring Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Service                    │
│  (Exposes /metrics endpoint with Prometheus format)│
└────────────┬────────────────────────────────────────┘
             │
             │ Scrapes every 5s
             ▼
┌─────────────────────────────────────────────────────┐
│                   Prometheus                         │
│  - Stores time-series metrics                       │
│  - Evaluates alert rules                            │
│  - Sends alerts to Alertmanager                     │
└────────────┬────────────────────────────────────────┘
             │
     ┌───────┴──────┬────────────────┐
     │              │                │
     ▼              ▼                ▼
┌─────────┐  ┌──────────┐  ┌────────────────┐
│ Grafana │  │ Alertmgr │  │ Drift Monitor  │
│Dashboards│  │  Alerts  │  │  (Custom)      │
└─────────┘  └──────────┘  └────────────────┘
```

---

## 2. Metrics Collection (Prometheus)

### Configuration (`prometheus.yml`)

```yaml
global:
  scrape_interval: 5s  # Collect metrics every 5 seconds

scrape_configs:
  - job_name: "fastapi"
    static_configs:
      - targets: ["fastapi:8000"]

rule_files:
  - /etc/prometheus/alert_rules.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]
```

### Metrics Exposed by FastAPI

| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total API requests (by endpoint, method) |
| `predictions_total` | Counter | Total predictions made (by model) |
| `request_latency_seconds` | Histogram | Request duration (buckets for P50, P95, P99) |
| `http_requests_total` | Counter | HTTP requests by status code |
| `up` | Gauge | Service health (1=up, 0=down) |

### Accessing Metrics

```bash
curl http://localhost:8000/metrics
```

Example output:
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST"} 1523.0

# HELP request_latency_seconds Latency of prediction endpoint
# TYPE request_latency_seconds histogram
request_latency_seconds_bucket{le="0.05"} 1200.0
request_latency_seconds_bucket{le="0.1"} 1480.0
request_latency_seconds_bucket{le="0.5"} 1520.0
```

---

## 3. Alerting Rules (`prometheus/alert_rules.yml`)

### Rule 1: High Request Latency

```yaml
- alert: HighRequestLatency
  expr: rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m]) > 0.5
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "High request latency on {{ $labels.instance }}"
    description: "Average latency is {{ $value }}s (threshold: 0.5s)"
```

**Trigger**: Average latency > 500ms for 1 minute

### Rule 2: High Error Rate

```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "High error rate on {{ $labels.instance }}"
    description: "Error rate is {{ $value }}% (threshold: 5%)"
```

**Trigger**: 5xx errors > 5% of total requests for 1 minute

### Rule 3: Instance Down

```yaml
- alert: InstanceDown
  expr: up == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Instance {{ $labels.instance }} down"
    description: "Service has been down for more than 1 minute"
```

**Trigger**: Service unreachable for 1 minute

---

## 4. Visualization (Grafana)

### Accessing Grafana

1. Navigate to http://localhost:3000
2. Login: `admin` / `admin`
3. Add Prometheus data source:
   - URL: `http://prometheus:9090`
   - Save & Test

### Dashboard Panels

#### Panel 1: Request Rate
```promql
rate(api_requests_total[5m])
```
**Visualization**: Time series line graph

#### Panel 2: Latency (P95)
```promql
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))
```
**Visualization**: Gauge (target: <100ms)

#### Panel 3: Error Rate
```promql
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100
```
**Visualization**: Graph with threshold line at 5%

#### Panel 4: Active Alerts
```promql
ALERTS{alertstate="firing"}
```
**Visualization**: Table

### Importing Dashboards

```bash
# Copy preconfigured dashboard
cp grafana/dashboards/api_monitoring.json /var/lib/grafana/dashboards/
```

---

## 5. Data Drift Detection

### Custom Drift Monitor (`monitoring/drift_monitor.py`)

#### Statistical Tests Used

**1. Kolmogorov-Smirnov (KS) Test**
- Tests if two distributions are different
- Null hypothesis: Distributions are the same
- Reject if p-value < 0.05

**2. Population Stability Index (PSI)**
```
PSI = Σ (actual% - expected%) * ln(actual% / expected%)
```
- PSI < 0.1: No drift
- 0.1 ≤ PSI < 0.2: Moderate drift
- PSI ≥ 0.2: Significant drift

#### Running Drift Detection

```bash
python monitoring/drift_monitor.py
```

**Output**: JSON report in `monitoring/reports/`

Example report:
```json
{
  "timestamp": "2024-12-24T12:50:34",
  "reference_size": 9707201,
  "current_size": 2426801,
  "features": {
    "PM10": {
      "ks_test": {
        "statistic": 0.082,
        "p_value": 0.001,
        "drift_detected": true
      },
      "psi": {
        "psi": 0.15,
        "drift_level": "moderate",
        "drift_detected": true
      },
      "drift_detected": true
    }
  }
}
```

#### Automated Drift Monitoring

Add to cron or GitHub Actions:
```yaml
- name: Daily Drift Check
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  run: python monitoring/drift_monitor.py
```

---

## 6. Model Lifecycle Management

### Lifecycle Stages

```
Development → Staging → Production → Archived
     ↓           ↓           ↓           ↓
  Testing    Validation  Monitoring  Retention
```

### Model Registry (`scripts/model_registry.py`)

#### Registering a Model

```python
from scripts.model_registry import ModelRegistry

registry = ModelRegistry()

version_id = registry.register_model(
    model_path="models/best_pm25_model.pkl",
    model_name="pm25_predictor",
    metrics={"rmse": 52.15, "mae": 23.24},
    stage="staging",
    tags=["xgboost", "v2"]
)
```

#### Promoting to Production

```python
registry.promote_model(
    model_name="pm25_predictor",
    version_id="v3",
    stage="production"
)
```

#### Loading Production Model

```python
model, metadata = registry.get_model(
    model_name="pm25_predictor",
    stage="production"
)
```

### Version History

```bash
# List all versions
python scripts/model_registry.py

# Output:
# v1 - Stage: staging   - 2024-12-01
# v2 - Stage: staging   - 2024-12-15
# v3 - Stage: production- 2024-12-24
```

---

## 7. Retraining Strategy

See `RETRAINING_PLAN.md` for detailed plan.

### Triggers

1. **Scheduled**: Monthly (1st of month)
2. **Performance Drift**: RMSE increases by >10%
3. **Data Drift**: PSI > 0.2 for key features
4. **Manual**: On-demand retraining

### Retraining Pipeline

```bash
# Triggered by GitHub Actions monthly
1. Fetch latest data
2. Run preprocessing
3. Train new model
4. Evaluate on validation set
5. Compare with current production model
6. If better: Register as staging
7. Manual review & promotion
```

### Evaluation Gates

Before promoting to production:
- ✅ RMSE ≤ Current production model
- ✅ Disparate Impact ∈ [0.8, 1.25]
- ✅ P95 latency < 100ms
- ✅ Manual sign-off

---

## 8. Logging System

### Application Logs (`main.py`)

```python
import logging

logger = logging.getLogger("audit")
logging.basicConfig(level=logging.INFO)

# Logs every request
logger.info(
    "method=%s path=%s status=%s duration_ms=%.2f",
    request.method,
    request.url.path,
    response.status_code,
    duration_ms
)
```

### Accessing Logs

```bash
# Docker logs
docker-compose logs -f fastapi

# Filter by severity
docker-compose logs fastapi | grep ERROR
```

### Log Aggregation (Optional)

For production, integrate with:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki** (Grafana Loki)
- **CloudWatch** (AWS)

---

## 9. Health Checks

### API Health Endpoint

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
```

### Kubernetes Liveness/Readiness (If using K8s)

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

---

## 10. KPIs and SLAs

### Defined SLAs

| Metric | Target | Current |
|--------|--------|---------|
| API Uptime | 99.5% | 99.8% |
| P95 Latency | <100ms | 85ms |
| Error Rate | <1% | 0.3% |
| Model RMSE | <55 µg/m³ | 52.15 µg/m³ |
| Drift Detection | Daily | ✅ |

### Monitoring Dashboard (Grafana)

Panels:
1. SLA compliance gauge
2. Request volume (last 24h)
3. Error timeline
4. Model performance trend

---

## 11. Incident Response

### Incident Workflow

1. **Alert fired** (Prometheus)
2. **Notification** (Alertmanager → Slack/Email)
3. **Triage** (Check Grafana dashboards)
4. **Mitigation** (Rollback or scale)
5. **Post-mortem** (Document in `docs/incidents/`)

### Rollback Procedure

```bash
# Via model registry
registry.promote_model("pm25_predictor", "v2", "production")

# Via Docker
docker-compose down
docker-compose up --build
```

---

## References

- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
- ML Monitoring Best Practices: https://ml-ops.org/content/phase-three
