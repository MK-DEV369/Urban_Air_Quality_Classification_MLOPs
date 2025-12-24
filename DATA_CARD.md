# Data Card: Urban Air Quality Dataset

## Dataset Overview
This dataset aggregates air quality measurements from multiple cities and stations to predict PM2.5 levels. It combines historical pollutant data with temporal features.

## Dataset Metadata
- **Name**: Master Air Quality Clean Dataset
- **Source**: Aggregated from various city-specific CSVs (e.g., AP001, DL001) located in `data/kaggle_csvs/`.
- **Version**: 1.0
- **License**: [Insert License Here, e.g., CC BY 4.0 or Open Database License]
- **Maintainers**: MLOps Project Team

## Data Composition
- **Total Records**: ~12 million (based on governance report)
- **Time Range**: [Start Date] to [End Date]
- **Granularity**: Hourly measurements

## Features
| Feature | Type | Description |
|---------|------|-------------|
| `PM2.5` | Float | Target variable. Particulate Matter < 2.5 micrometers. |
| `PM10` | Float | Particulate Matter < 10 micrometers. |
| `O3` | Float | Ozone concentration. |
| `CO` | Float | Carbon Monoxide concentration. |
| `Timestamp` | DateTime | Date and time of measurement. |
| `hour` | Integer | Hour of the day (0-23). |
| `dayofweek` | Integer | Day of the week (0=Monday, 6=Sunday). |
| `month` | Integer | Month of the year (1-12). |

## Data Cleaning & Preprocessing
1. **Missing Values**: Rows with missing `Timestamp` or `PM2.5` are dropped. Other features (`PM10`, `O3`, `CO`) are imputed using the median.
2. **Type Conversion**: Pollutant columns are converted to numeric, coercing errors to NaN.
3. **Feature Engineering**: Temporal features (`hour`, `dayofweek`, `month`) are extracted from `Timestamp`.

## Known Limitations & Biases
- **Geographic Bias**: Data may be heavily weighted towards specific cities (e.g., Delhi) depending on the number of stations.
- **Temporal Gaps**: Some stations may have missing data for extended periods.
- **Sensor Error**: Raw sensor data may contain outliers or calibration errors.

## Governance & Compliance
- **PII**: No Personally Identifiable Information is contained in this dataset.
- **Fairness**: Evaluated for bias across temporal groups (e.g., Weekend vs. Weekday). See `governance_report.json`.
