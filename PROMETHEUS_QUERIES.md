# Prometheus Query Examples (PromQL) - MLOps Testing

## Basic Setup
- **Prometheus URL**: http://localhost:9090
- **Query Endpoint**: http://localhost:9090/api/v1/query
- **Range Query**: http://localhost:9090/api/v1/query_range

---

## 1. Health & Status Checks

### Service Up Status
```promql
up
```
**What it does**: Shows which targets are up (1) or down (0)
**Use case**: Verify FastAPI and other services are running

### All Available Metrics
```promql
{__name__=~".+"}
```
**What it does**: Lists all available metrics in Prometheus
**Use case**: Discover what metrics are being collected

---

## 2. API Request Metrics

### Total API Requests
```promql
api_requests_total
```
**What it does**: Total count of all API requests
**Use case**: Overall traffic volume

### Requests Per Endpoint
```promql
api_requests_total{endpoint=~"/predict|/metrics|/"}
```
**What it does**: Filter requests by specific endpoints
**Use case**: Identify which endpoints receive traffic

### Request Rate (Last 5 Minutes)
```promql
rate(api_requests_total[5m])
```
**What it does**: Requests per second over 5-minute window
**Use case**: Monitor traffic rate trends

### Requests Per Method (GET vs POST)
```promql
sum(rate(api_requests_total[5m])) by (method)
```
**What it does**: Group request rate by HTTP method
**Use case**: Compare GET vs POST traffic

### Requests Per Endpoint (Breakdown)
```promql
sum(rate(api_requests_total[5m])) by (endpoint)
```
**What it does**: Which endpoints get most traffic
**Use case**: Identify bottlenecks or heavy endpoints

---

## 3. Prediction Metrics

### Total Predictions Made
```promql
predictions_total
```
**What it does**: Count of all predictions (by model)
**Use case**: Model usage tracking

### Predictions Per Model
```promql
sum(predictions_total) by (model_name)
```
**What it does**: Breakdown by model type (XGBRegressor, RandomForest, etc.)
**Use case**: Compare model usage

### Prediction Rate
```promql
rate(predictions_total[5m])
```
**What it does**: Predictions per second
**Use case**: Monitor prediction throughput

---

## 4. Latency & Performance

### Average Latency (Last 5 Minutes)
```promql
rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m])
```
**What it does**: Average request time in seconds
**Use case**: Monitor API response time

### Latency Percentiles (95th percentile)
```promql
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))
```
**What it does**: 95% of requests complete in this time (seconds)
**Use case**: SLA compliance - max acceptable time

### Latency Percentiles (99th percentile - high outliers)
```promql
histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m]))
```
**What it does**: Maximum latency for 99% of requests
**Use case**: Identify worst-case performance

### Latency Percentiles (50th percentile - median)
```promql
histogram_quantile(0.50, rate(request_latency_seconds_bucket[5m]))
```
**What it does**: Median request latency
**Use case**: Typical user experience

### Latency by Endpoint
```promql
histogram_quantile(0.95, sum(rate(request_latency_seconds_bucket[5m])) by (le, endpoint))
```
**What it does**: 95th percentile latency per endpoint
**Use case**: Identify slow endpoints

---

## 5. Combined Metrics (Advanced)

### API Health Score (Requests × Latency)
```promql
(rate(api_requests_total[5m]) * histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m])))
```
**What it does**: Traffic × Latency (combined metric)
**Use case**: Overall API health indicator

### Error Rate (4xx + 5xx responses)
```promql
rate(api_requests_total{status=~"4..|5.."}[5m])
```
**What it does**: Request failure rate
**Use case**: Monitor API errors (if status label added)

### Prediction Accuracy Proxy (Successful requests)
```promql
rate(api_requests_total{endpoint="/predict"}[5m])
```
**What it does**: Prediction endpoint traffic
**Use case**: Monitor prediction service load

---

## 6. Time-Series Analysis

### Request Count Over 1 Hour
```promql
increase(api_requests_total[1h])
```
**What it does**: Total requests in past hour
**Use case**: Hourly traffic patterns

### Latency Trend (1-Hour Window)
```promql
rate(request_latency_seconds_sum[1h]) / rate(request_latency_seconds_count[1h])
```
**What it does**: Average latency trend
**Use case**: Performance degradation detection

### Moving Average (5-Minute rolling)
```promql
avg_over_time(rate(api_requests_total[5m])[30m:1m])
```
**What it does**: Smoothed request rate over 30 minutes
**Use case**: Filter noise from rate metrics

---

## 7. Python Client Metrics

### Python GC (Garbage Collection) Activity
```promql
rate(python_gc_objects_collected_total[5m])
```
**What it does**: Garbage collection activity
**Use case**: Memory pressure indicator

### Python GC Collections Per Generation
```promql
sum(rate(python_gc_objects_collected_total[5m])) by (generation)
```
**What it does**: Breakdown by generation (0, 1, 2)
**Use case**: Memory generation analysis

---

## 8. Testing Queries (For Load Testing)

### Requests Over Last 30 Minutes
```promql
sum(rate(api_requests_total[30m]))
```
**What it does**: Average request rate (30-min window)
**Use case**: Baseline before load test

### Compare Before/After Load Test
```promql
rate(api_requests_total[5m]) > 10
```
**What it does**: Show metrics where rate > 10 req/sec
**Use case**: Identify load test impact

### Latency Spike Detection
```promql
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m])) > 0.1
```
**What it does**: Alert if 95th percentile latency > 100ms
**Use case**: Performance regression detection

---

## 9. Service Health Checks

### Service Up (Binary)
```promql
up{job="fastapi"}
```
**What it does**: Is FastAPI service running? (1=yes, 0=no)
**Use case**: Basic availability check

### Prometheus Scrape Success Rate
```promql
rate(prometheus_sd_linode_api_call_duration_seconds_count[5m])
```
**What it does**: Prometheus scrape frequency
**Use case**: Verify metrics collection

### Service Restart Count
```promql
increase(process_start_time_seconds[5m])
```
**What it does**: Detect process restarts
**Use case**: Stability monitoring

---

## 10. Custom Query Examples

### Total API Calls by Method & Endpoint
```promql
sum by (method, endpoint) (rate(api_requests_total[5m]))
```

### Top 5 Slowest Endpoints
```promql
topk(5, histogram_quantile(0.95, sum(rate(request_latency_seconds_bucket[5m])) by (endpoint)))
```

### Request Rate Growth (Rate of change)
```promql
rate(rate(api_requests_total[5m])[1m])
```

### SLA Compliance (% under 100ms)
```promql
100 * (rate(request_latency_seconds_bucket{le="0.1"}[5m]) / rate(request_latency_seconds_count[5m]))
```

---

## Testing Script (PowerShell)

```powershell
# Function to query Prometheus
function Query-Prometheus {
    param(
        [string]$Query,
        [string]$Server = "http://localhost:9090"
    )
    
    $url = "$Server/api/v1/query?query=$(([System.Uri]::EscapeDataString($Query)))"
    
    try {
        $response = Invoke-RestMethod -Uri $url -Method Get
        return $response.data.result
    } catch {
        Write-Error "Query failed: $_"
    }
}

# Example: Get current request rate
$result = Query-Prometheus "rate(api_requests_total[5m])"
Write-Host "Request Rate:" 
$result | Format-Table

# Example: Get latency
$result = Query-Prometheus 'histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))'
Write-Host "95th Percentile Latency:" 
$result | Format-Table
```

---

## Testing Workflow

### Step 1: Baseline (No Load)
```promql
rate(api_requests_total[5m])  # Should be 0 or near 0
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))  # Should be < 50ms
```

### Step 2: Generate Load
```powershell
# Run load test in parallel
1..100 | ForEach-Object -Parallel {
    $payload = @{
        "PM10" = 100 + (Get-Random -Maximum 100)
        "O3" = 30 + (Get-Random -Maximum 50)
        "CO" = 1.0 + (Get-Random)
        "hour" = Get-Random -Maximum 24
        "dayofweek" = Get-Random -Maximum 7
        "month" = Get-Random -Minimum 1 -Maximum 13
    } | ConvertTo-Json
    
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
        -Body $payload -ContentType "application/json"
} -ThrottleLimit 10
```

### Step 3: Monitor During Load
```promql
rate(api_requests_total[5m])  # Should increase significantly
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))  # May increase
predictions_total  # Should increase
```

### Step 4: Post-Load Analysis
```promql
rate(api_requests_total[5m])  # Back to baseline
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))  # Should normalize
increase(api_requests_total[1h])  # Total requests in last hour
```

---

## Expected Results (Healthy API)

| Metric | Threshold | Status |
|--------|-----------|--------|
| **Request Rate** | 0-100 req/sec | ✅ Normal |
| **P95 Latency** | < 100ms | ✅ Good |
| **P99 Latency** | < 500ms | ✅ Good |
| **Service Up** | 1 (running) | ✅ Up |
| **Error Rate** | < 1% | ✅ Low |
| **Python GC** | < 1 collections/sec | ✅ Normal |

---

## Grafana Integration

Once queries are tested in Prometheus, create Grafana dashboard:

1. **Panel 1**: Request Rate
   - Query: `rate(api_requests_total[5m])`
   - Type: Graph
   - Unit: req/s

2. **Panel 2**: P95 Latency
   - Query: `histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))`
   - Type: Graph
   - Unit: seconds

3. **Panel 3**: Total Requests
   - Query: `api_requests_total`
   - Type: Stat (single value)

4. **Panel 4**: Predictions Per Model
   - Query: `sum(predictions_total) by (model_name)`
   - Type: Pie Chart

---

## Troubleshooting Queries

### No Results?
```promql
# Check if metrics exist
{__name__=~"api_.*"}

# Check time range
api_requests_total[1h]  # Look back 1 hour instead of default 5m
```

### Wrong Data Type?
```promql
# Verify metric type
REGISTRY  # Shows all registered metrics

# Check specific metric
api_requests_total  # Should show Counter type
request_latency_seconds_bucket  # Should show Histogram
```

### Need to Debug?
```promql
# Show raw values
api_requests_total

# Show with labels
api_requests_total{endpoint="/predict"}

# Show rate of change
rate(api_requests_total[1m])
```

---

## Quick Reference Card

| Task | Query |
|------|-------|
| Service status | `up` |
| Total requests | `api_requests_total` |
| Request rate | `rate(api_requests_total[5m])` |
| Avg latency | `rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m])` |
| P95 latency | `histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))` |
| Predictions | `predictions_total` |
| Requests/endpoint | `sum(rate(api_requests_total[5m])) by (endpoint)` |
| Requests/method | `sum(rate(api_requests_total[5m])) by (method)` |

---

Copy these queries directly into Prometheus web UI (http://localhost:9090) and test them!
