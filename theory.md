Here's the core statistical theory, in the order you'll encounter it:                                                                                                                                   
                                                                                                                                                                                                          
  ---                                                                                                                                                                                                     
  1. Stationarity — the foundation of everything                                                                                                                                                                                                                                                                                                                                                                    
  A time series is stationary if its mean, variance, and autocorrelation structure don't change over time. Most economic series (prices, GDP, rates) are not stationary — they trend upward.              
                                                        
  Why it matters: almost every model assumes stationarity. A regression between two trending series will look significant even if they're completely unrelated ("spurious regression").

  Solution: take differences. If the original series $y_t$ is non-stationary but $\Delta y_t = y_t - y_{t-1}$ is stationary, we say $y_t$ is integrated of order 1, written $I(1)$.

  ---
  2. Unit root tests — detecting non-stationarity

  Two tests that ask complementary questions:

  - ADF (Augmented Dickey-Fuller): $H_0$ = the series has a unit root (is non-stationary). Reject → stationary. Low p-value is good.
  - KPSS: $H_0$ = the series is stationary. Reject → non-stationary. Low p-value is bad.

  They're used together because each has low power alone. The decision rule: call the series stationary only if ADF rejects (p < 0.05) AND KPSS does not reject (p ≥ 0.05). If they disagree, difference  
  the series.

  ---
  3. SARIMA — the App 1 model

  ARIMA(p, d, q) stands for:
  - d = how many times you need to difference to get stationarity
  - p = how many past values of the series predict the next one (AR terms)
  - q = how many past errors predict the next one (MA terms)

  SARIMA(p, d, q)(P, D, Q)[12] adds seasonal versions of the same thing at lag 12 (months). D=1 means one seasonal difference ($y_t - y_{t-12}$), which removes annual seasonality.

  Box-Jenkins methodology is the 4-step process to build this:
  1. Identify d and D (unit root tests + seasonal ACF)
  2. Estimate — grid search over p, q, P, Q; pick lowest AIC
  3. Diagnose residuals — are they white noise? (Ljung-Box, Jarque-Bera, ADF)
  4. Forecast — project forward and transform back to original scale

  AIC (Akaike Information Criterion): penalises model complexity. Lower = better fit without overfitting.

  Ljung-Box test: checks whether residual autocorrelations are all zero (white noise). If residuals still have autocorrelation, the model didn't capture all the structure — it's technically invalid.    

  ---
  4. Cointegration — when I(1) series move together long-run

  If two I(1) series are cointegrated, there exists a linear combination of them that is stationary — a "long-run equilibrium." They can diverge short-term but always return to each other.

  Example: prices and wages both trend up, but the ratio (real wage) is stable.

  Johansen trace test determines how many such equilibrium relationships ("cointegrating vectors") exist in a system of $k$ variables. It tests sequentially: $r=0$? $r≤1$? etc. We found $r=1$ — exactly 
  one stable long-run relationship among HICP, interest rate, and industrial production.

  ---
  5. VECM vs VAR — the App 2 choice

  VAR (Vector Autoregression): each variable is regressed on its own lags and the lags of all other variables. Simple, symmetric, no assumptions about causality. But if variables are I(1), VAR in levels
   gives spurious results.

  VECM (Vector Error Correction Model): used when variables are cointegrated. It models first differences (so stationarity is satisfied), but adds an "error correction term" — the deviation from the    
  long-run equilibrium — as an extra predictor. The loading coefficient α measures how fast each variable corrects back.

  In our project: α for the interest rate is −1.268 and highly significant — the interest rate adjusts fast when prices deviate from equilibrium. α for industrial production is −0.006 and insignificant 
  — output doesn't respond at monthly frequency.
    ---
  6. Granger causality

  "X Granger-causes Y" means: past values of X help predict Y, beyond what Y's own history already tells you.

  It's not true causality — it's predictive precedence. Tested with an F-test (or chi-squared). Must be run on stationary series (first differences for I(1) variables) to have valid inference.

  In the project: only interest rate → HICP is significant (p = 0.039). This says rate changes have predictive power over future inflation, consistent with monetary policy theory.

  ---
  7. Impulse Response Functions (IRF)

  From the estimated VECM, you can simulate: "what happens to variable Y over the next 24 months if variable X receives a one-standard-deviation shock today, holding everything else equal?"

  The answer is the IRF. It shows the dynamic transmission mechanism — we see that a positive interest rate shock gradually dampens consumer prices, consistent with contractionary monetary policy       
  reducing inflation over 12–24 months.

  ---
  The logical flow of the project: check stationarity → difference to achieve it (App 1: one series) → build SARIMA, forecast → check if multiple I(1) series share a long-run equilibrium → if yes, use  
  VECM → test short-run Granger causality → trace dynamic responses via IRF.
