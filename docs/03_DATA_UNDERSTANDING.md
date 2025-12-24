# Data Understanding and Feature Engineering

## 1. Data Collection

### Data Sources
- **Kaggle Dataset**: Air Quality Data in India (2015-2020)
- **Source Files**: 453 CSV files representing different monitoring stations across Indian cities
- **Total Records**: ~14 million observations
- **Time Period**: 2015-2020

### Station Distribution
- Delhi (DL): 29 stations
- Bihar (BR): 35 stations
- Chhattisgarh (CG): 14 stations
- Andhra Pradesh (AP): 10 stations
- Other states: Variable coverage

## 2. Data Exploration

### Raw Features (90+ columns)
The raw dataset contains numerous pollutant measurements and meteorological parameters:

**Pollutants:**
- PM2.5, PM10 (Particulate Matter)
- NO, NO2, NOx (Nitrogen oxides)
- SO2 (Sulfur dioxide)
- CO (Carbon monoxide)
- O3 (Ozone)
- NH3 (Ammonia)
- Benzene, Toluene, Xylene (Volatile Organic Compounds)

**Meteorological:**
- Temperature (Temp, AT)
- Relative Humidity (RH)
- Wind Speed (WS) and Direction (WD)
- Solar Radiation (SR)
- Barometric Pressure (BP)
- Rainfall (RF)

**Temporal:**
- From Date, To Date (measurement period)

### Data Quality Issues

1. **Missing Values**: Significant missingness in many features (~30-60%)
2. **Duplicate Columns**: Same measurement with different units (e.g., `Temp (degree C)`, `Temp ()`, `Temp (ug/m3)`)
3. **Inconsistent Units**: Mixed units across stations
4. **Outliers**: Sensor errors producing extreme values
5. **Temporal Gaps**: Some stations have long periods without data

## 3. Data Profiling

### Target Variable: PM2.5
- **Mean**: 89.3 µg/m³
- **Median**: 60.1 µg/m³
- **Std Dev**: 82.5 µg/m³
- **Range**: 0 - 999 µg/m³ (contains outliers)
- **Missing**: ~15%

### Key Predictors
| Feature | Correlation with PM2.5 | Missing % |
|---------|------------------------|-----------|
| PM10    | 0.87                   | 18%       |
| CO      | 0.42                   | 35%       |
| NO2     | 0.38                   | 40%       |
| O3      | -0.15                  | 30%       |

## 4. Data Cleaning and Preprocessing

### Implemented in `scripts/data_preprocessing.py`

#### Step 1: Timestamp Parsing
```python
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df.dropna(subset=["Timestamp"], inplace=True)
```

#### Step 2: Feature Engineering (Temporal)
```python
df["hour"] = df["Timestamp"].dt.hour
df["dayofweek"] = df["Timestamp"].dt.dayofweek  # 0=Monday
df["month"] = df["Timestamp"].dt.month
df["year"] = df["Timestamp"].dt.year
```

#### Step 3: Type Conversion
```python
for col in ["PM2.5", "PM10", "O3", "CO"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

#### Step 4: Handle Missing Values
- **Target (PM2.5)**: Drop rows with missing target
- **Features**: Impute with median
```python
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
```

#### Step 5: Outlier Removal
Remove extreme outliers using 1st and 99th percentiles:
```python
Q1 = df["PM2.5"].quantile(0.01)
Q3 = df["PM2.5"].quantile(0.99)
df = df[(df["PM2.5"] >= Q1) & (df["PM2.5"] <= Q3)]
```

### Final Dataset
- **Records**: ~12 million (after cleaning)
- **Features Used**: 6 (PM10, O3, CO, hour, dayofweek, month)
- **Reduction**: ~15% data removed (missing values + outliers)

## 5. Feature Engineering

### Selected Features

#### Pollutant Features
1. **PM10**: Strong predictor (0.87 correlation)
2. **O3**: Negative correlation (-0.15) - important for completeness
3. **CO**: Moderate correlation (0.42)

#### Temporal Features
4. **hour**: Captures diurnal patterns (traffic peaks, industrial activity)
5. **dayofweek**: Weekend vs. weekday patterns
6. **month**: Seasonal trends (winter pollution, monsoon washout)

### Feature Selection Rationale

**Why these 6 features?**
1. **Availability**: Low missing data after imputation
2. **Correlation**: Strong predictive power for PM2.5
3. **Interpretability**: Easy to explain to stakeholders
4. **Operational**: Can be collected in real-time

**Features Excluded:**
- Meteorological variables (Temperature, Humidity): High missingness (>50%)
- VOCs (Benzene, Toluene): Limited availability across stations
- Other pollutants (NO2, SO2): High missingness and redundancy

## 6. Exploratory Data Analysis (EDA)

### Temporal Patterns
- **Daily**: PM2.5 peaks during morning (7-9 AM) and evening (6-8 PM) rush hours
- **Weekly**: Slightly lower on weekends (~5% reduction)
- **Seasonal**: Higher in winter (November-February), lower during monsoon (July-September)

### Spatial Patterns
- **Delhi**: Highest average PM2.5 (~120 µg/m³)
- **Coastal Cities**: Lower PM2.5 (~60 µg/m³)
- **Industrial Zones**: Elevated CO and PM10

### Correlations
```
PM2.5 correlations:
  PM10:      0.87 (very strong)
  CO:        0.42 (moderate)
  NO2:       0.38 (moderate)
  O3:       -0.15 (weak negative)
  hour:      0.12 (weak)
  month:     0.08 (weak)
```

## 7. Data Versioning and Governance

### Version Control Strategy
- **Raw Data**: Stored in `data/kaggle_csvs/` (not tracked by git due to size)
- **Processed Data**: `data/master_airquality_clean.csv` (tracked via git)
- **Version Tags**: Date-based naming (e.g., `data_v2024_12_24`)

### Governance (See `DATA_CARD.md`)
- **Licensing**: Open Dataset (Kaggle - check specific license)
- **PII**: No personally identifiable information
- **Bias**: Evaluated for temporal bias (weekend vs. weekday)
- **Quality Checks**: Automated validation in preprocessing pipeline

## 8. Data Transformation Pipeline

The complete data pipeline is orchestrated via `scripts/pipeline.py`:

1. **Ingestion**: `scripts/data_ingestion.py` - Load all CSVs
2. **Preprocessing**: `scripts/data_preprocessing.py` - Clean and transform
3. **Output**: `data/master_airquality_clean.csv`

### Running the Pipeline
```bash
python scripts/data_ingestion.py      # Step 1
python scripts/data_preprocessing.py  # Step 2
```

## 9. Feature Importance (Post-Training)

After training, feature importance was analyzed using SHAP:

| Feature   | Importance | Interpretation |
|-----------|------------|----------------|
| PM10      | 0.65       | Primary predictor |
| CO        | 0.15       | Industrial activity marker |
| O3        | 0.10       | Photochemical reactions |
| hour      | 0.06       | Traffic patterns |
| month     | 0.03       | Seasonal effects |
| dayofweek | 0.01       | Minimal impact |

See `artifacts/shap_summary.png` for visualization.

## 10. References

- Central Pollution Control Board (CPCB) India
- National Air Quality Index (AQI) Standards
- WHO Air Quality Guidelines
