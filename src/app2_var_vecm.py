import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


DATA_PATH = "../data/app2_monthly.csv"
FIG_DIR = "../figures"
OUTPUT_DIR = "../output"
MAX_LAGS = 6
IRF_PERIODS = 24


def adf_pvalue(series: pd.Series) -> float:
    return adfuller(series.dropna(), autolag="AIC")[1]


def adf_report(df: pd.DataFrame, columns: List[str]) -> str:
    lines = ["ADF Unit Root Tests", "-" * 50]
    for col in columns:
        p_level = adf_pvalue(df[col])
        p_diff = adf_pvalue(df[col].diff().dropna())
        lines.append(
            f"  {col:12s}  level p={p_level:.4f}  Δ1 p={p_diff:.4f}  "
            f"→ I({'0' if p_level < 0.05 else '1'})"
        )
    return "\n".join(lines)


def granger_report(df: pd.DataFrame, max_lag: int = MAX_LAGS) -> str:
    lines = ["\nGranger Causality Tests on first differences (min p-value across lags 1–" + str(max_lag) + ")", "-" * 50]
    for y in df.columns:
        for x in df.columns:
            if x == y:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(df[[y, x]], maxlag=max_lag, verbose=False)
                min_p = min(res[lag][0]["ssr_ftest"][1] for lag in res)
                sig = "**" if min_p < 0.05 else "  "
                lines.append(f"  {sig} {x} → {y}:  min p = {min_p:.4f}")
            except Exception as exc:
                lines.append(f"     {x} → {y}:  error ({exc})")
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["time"]).dropna().sort_values("time")
    df = df.set_index("time").asfreq("MS").dropna()

    columns = ["hicp", "short_rate", "indprod"]
    df = df[columns]

    # Log-transform price levels; interest rate stays in % points
    log_df = df.copy()
    log_df["hicp"] = np.log(df["hicp"])
    log_df["indprod"] = np.log(df["indprod"])

    print(f"Sample: {log_df.index[0].date()} – {log_df.index[-1].date()}  ({len(log_df)} obs)\n")

    # ── 1. Unit root tests ───────────────────────────────────────────────────
    adf_text = adf_report(log_df, columns)
    print(adf_text)

    # ── 2. Johansen cointegration ────────────────────────────────────────────
    # Determine AIC-optimal lag order first so it can be used consistently
    # for both the Johansen test and the subsequent VECM/VAR estimation.
    lag_order = max(int(VAR(log_df.diff().dropna()).select_order(MAX_LAGS).aic), 1)

    johansen = coint_johansen(log_df, det_order=0, k_ar_diff=lag_order)
    trace_stat = johansen.lr1
    crit_5 = johansen.cvt[:, 1]
    coint_rank = int(np.sum(trace_stat > crit_5))

    johansen_lines = ["\nJohansen Cointegration Test (trace, 5%)", "-" * 50]
    johansen_lines.append(f"  Lag order (AIC): {lag_order}")
    for i, (ts, cv) in enumerate(zip(trace_stat, crit_5)):
        johansen_lines.append(f"  H0: rank <= {i}   trace={ts:.3f}  cv={cv:.3f}  {'reject' if ts > cv else 'fail to reject'}")
    johansen_lines.append(f"\n  Cointegration rank: {coint_rank}")
    johansen_text = "\n".join(johansen_lines)
    print(johansen_text)

    # ── 3. Fit VAR or VECM ───────────────────────────────────────────────────
    if coint_rank > 0:
        model = VECM(log_df, k_ar_diff=lag_order, coint_rank=coint_rank, deterministic="co")
        result = model.fit()
        model_text = result.summary().as_text()
        with open(os.path.join(OUTPUT_DIR, "app2_vecm_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write(model_text)
        irf = result.irf(IRF_PERIODS)
        model_name = f"VECM (rank={coint_rank}, k_ar_diff={lag_order})"
    else:
        diff_df = log_df.diff().dropna()
        result = VAR(diff_df).fit(lag_order)
        model_text = result.summary().as_text()
        with open(os.path.join(OUTPUT_DIR, "app2_var_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write(model_text)
        irf = result.irf(IRF_PERIODS)
        model_name = f"VAR (lags={lag_order})"

    print(f"\nFitted: {model_name}")

    # ── 4. Impulse Response Functions ────────────────────────────────────────
    irf.plot(orth=True, figsize=(12, 8))
    plt.suptitle(f"Orthogonalized IRF – {model_name}", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app2_irf.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Granger causality ─────────────────────────────────────────────────
    granger_text = granger_report(log_df.diff().dropna())
    print(granger_text)

    # ── 6. Save full report ──────────────────────────────────────────────────
    report = "\n".join([adf_text, johansen_text, granger_text])
    with open(os.path.join(OUTPUT_DIR, "app2_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)


if __name__ == "__main__":
    main()
