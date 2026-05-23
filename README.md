# time-series-project

This repository contains a two-part Time Series project focused on Romania:

- Application 1: EUR/RON exchange rate forecasting with ARIMA.
- Application 2: Multivariate analysis using EUR/RON, HICP inflation, and a short-term interest rate.

## Data sources

The scripts pull data from Eurostat using its public API. The dataset codes and filters are defined in
[src/config.py](src/config.py).

## Setup

1) Create and activate a Python environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Fetch data

```bash
python src/fetch_data.py
```

This writes CSV files to the data folder and builds a merged dataset for Application 2.

## Run Application 1 (ARIMA)

```bash
python src/app1_arima.py
```

Outputs:
- figures/app1_series_log.png
- figures/app1_acf_pacf.png
- figures/app1_forecast.png
- output/app1_diagnostics.txt
- output/app1_metrics.txt

## Run Application 2 (VAR/VECM)

```bash
python src/app2_var_vecm.py
```

Outputs:
- figures/app2_irf.png
- output/app2_var_summary.txt or output/app2_vecm_summary.txt

## Notes

If a dataset filter key is not available in Eurostat, the downloader prints a warning and skips that filter.
In that case, adjust the filters in [src/config.py](src/config.py).