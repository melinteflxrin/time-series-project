# time-series-project

This repository contains a two-part Time Series project focused on Romania.

- Application 1: EUR/RON exchange rate forecasting with ARIMA (monthly).
- Application 2: Multivariate analysis using EUR/RON, HICP inflation, and a short-term interest rate (monthly).

## Data sources

The scripts pull data from Eurostat using its public API. The dataset codes and filters are defined in
[src/config.py](src/config.py). The downloader normalizes wide Eurostat tables and selects the most complete
series when multiple variants exist.

Current datasets (Eurostat):
- Exchange rate: `ert_bil_eur_m` (EUR/RON, monthly).
- Inflation: `prc_hicp_midx` (HICP, CP00, index 2015=100).
- Short-term rate: `irt_st_m` (short-term interest rate).

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

Modeling notes:
- Uses log of EUR/RON and enforces at least one difference (d>=1).
- Drift is included when differencing is used.
- Forecast horizon is 6 months to keep uncertainty reasonable.
- Monthly frequency is enforced and missing months are dropped.

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

Notes:
- The script prints Granger causality tests for all pairwise directions and lags 1..6.
- The console output is verbose by default; keep it for the report interpretation.

## Collaboration notes

- Data CSVs are written to the data folder and can be reused without re-downloading.
- Figures and metrics are in the figures and output folders; keep them for the report.
- If the results look too flat or noisy, check the ARIMA order line printed in the console.

## Notes

If a dataset filter key is not available in Eurostat, the downloader prints a warning and skips that filter.
In that case, adjust the filters in [src/config.py](src/config.py).