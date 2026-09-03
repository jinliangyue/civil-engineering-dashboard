# Interview Talking Points · Machine Learning Section

> This document is the **current** interview talking script aligned with the P0.5 / P0.6 experimental results.
>
> Historical context (the period when this project used manually-estimated fallback data) is preserved at the end as **deprecated background**, clearly marked.

---

## Project Name

**China Industrial PPI Time-Series Analysis and Forecasting Platform**

---

## Project Narrative (current)

```
China Industrial PPI
       ↓
Official monthly data via akshare
       ↓
132 observations (2015-01 ~ 2025-12)
       ↓
Train / Validation / Final Test split (84 / 24 / 24 months)
       ↓
4 statistical baselines (Naive / Seasonal Naive / MA / SES)
       ↓
Prophet · XGBoost · PyTorch LSTM
       ↓
Validation-weighted Ensemble
       ↓
Final OOS Test
       ↓
Walk-forward Validation
```

---

## 5-Minute Compact Version

This project's machine-learning module is the second highlight of my Streamlit dashboard.

The data: 132 monthly PPI observations (2015-01 to 2025-12), retrieved through akshare from China's National Bureau of Statistics — not simulated, not manually estimated.

I evaluated 7 models:

- 4 statistical baselines (Naive, Seasonal Naive, MA, SES)
- Prophet (additive trend + yearly seasonality)
- XGBoost (gradient boosting on hand-crafted causal features)
- PyTorch LSTM (2-layer, rolling one-step-ahead)

I implemented 15 hand-crafted causal features for XGBoost (lag 1/3/6/12 + rolling mean/std 3/6/12 + YoY + MoM), all using `shift(1)` to ensure they only depend on past values.

LSTM hyperparameters were tuned by grid search restricted to the Train segment only (P0.3 commit `42ae111`).

I built a validation-weighted ensemble: weights = 1 / Validation MAPE, normalized to sum to 1, then locked before final Test evaluation.

Final Test results (2024-01 to 2025-12, 24 months strict OOS):

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| XGBoost | 0.3482 | 0.4824 | 0.3558% | 0.5209 |
| LSTM | 0.4288 | 0.5224 | 0.4387% | 0.4381 |
| Ensemble | 0.3473 | 0.4589 | 0.3551% | 0.5664 |

**Important interpretation note**: The 2024-2025 Final Test period was relatively low-volatility (annual range 1.7-2.1). The Naive baseline already achieved MAPE = 0.3667%. Therefore the 0.3551% Ensemble MAPE should not be interpreted as universal forecasting accuracy — it mainly reflects this regime's stability.

Walk-forward Validation (3 expanding-window folds over 2021/2022/2023):

| Model | F1 (2021) | F2 (2022) | F3 (2023) | Mean | Std |
|---|---:|---:|---:|---:|---:|
| Naive | 1.35% | 0.98% | 0.72% | 1.02% | 0.26% |
| XGBoost | 2.86% | 1.03% | 0.89% | 1.60% | 0.90% |
| LSTM | 1.56% | 1.94% | 0.73% | 1.41% | 0.50% |

---

## 15-Minute Deep Version

### Background (2 minutes)

China's PPI is the key benchmark for material price adjustment in engineering cost contracts. Public PPI data is available but only as monthly bulletins, without convenient cross-industry comparison tools.

I built an end-to-end monthly time-series forecasting pipeline using 132 real monthly PPI observations (akshare → National Bureau of Statistics, 2015-01 to 2025-12).

I implemented strict data isolation:

- Train 84 months (2015-01 to 2021-12)
- Validation 24 months (2022-01 to 2023-12) — used for hyperparameter selection and ensemble weight computation only
- Final Test 24 months (2024-01 to 2025-12) — used exactly once for final reporting

### Feature Engineering (4 minutes)

15 causal features, all using `shift(1)`:

- **Lag features**: `lag_1`, `lag_3`, `lag_6`, `lag_12` (price at t-1, t-3, t-6, t-12)
- **Rolling features**: `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_12`, `rolling_std_3`, `rolling_std_6`, `rolling_std_12` (computed on `shift(1)` so they only see past values)
- **Time features**: `year`, `month`, `quarter`
- **Derived features**: `yoy_change` (shift(1) − shift(13)), `mom_change` (shift(1) − shift(2))

All features are causal — no future information leakage.

### Model Training (3 minutes)

**XGBoost**:
- XGBRegressor (n_estimators=200, max_depth=4, learning_rate=0.05)
- Rolling one-step-ahead prediction on Test: each step uses features computed from history (which includes the actual values up to that point), then appends the actual after prediction
- causal features prevent target leakage that would otherwise occur from rolling features that include the current target

**LSTM** (PyTorch):
- 2-layer LSTM with hidden_size=32, dropout=0.1, seq_length=6, num_layers=2, lr=0.001 (P0.3 grid search result on Train only)
- StandardScaler fit only on training data
- Rolling one-step-ahead on Test, starting with the last 6 points of training as the first input

**Prophet**:
- Additive trend + yearly seasonality
- Fit on Train, predict on Test (out-of-sample)

### Ensemble Construction (2 minutes)

Validation-weighted ensemble:

```
raw_weight_i = 1 / Validation_MAPE_i
weight_i = raw_weight_i / sum(raw_weights)
```

Validation-derived weights:

| Model | Weight |
|---|---:|
| Naive | 0.28189 |
| Seasonal Naive | 0.03278 |
| MA | 0.15227 |
| SES | 0.11433 |
| Prophet | 0.00947 |
| XGBoost | 0.24592 |
| LSTM | 0.16333 |

Weights are locked before Final Test evaluation. Test set plays no role in weight computation.

### Walk-forward Validation (1 minute)

3 expanding-window folds (F1 = 72 train → 12 test, F2 = 84 → 12, F3 = 96 → 12). Folds cover 2021 / 2022 / 2023. Walk-forward ensemble weights are not recomputed (the validation period is fixed).

### Limitations (1 minute)

Three honest limitations:

1. **Sample size**: Only 132 monthly observations; walk-forward folds contain only 12 test points each.
2. **No exogenous variables**: PPI is a univariate series; no PMI / CPI / energy prices / FX are included.
3. **2024-2025 regime caveat**: Final Test is a low-volatility period; low MAPE numbers should be interpreted alongside the Naive baseline.

---

## Q&A Preparation

Q1: Why did you choose PPI as the prediction target?
A: PPI is the official industrial price benchmark published by China's National Bureau of Statistics, and it directly drives the material-price adjustment formula in engineering cost contracts.

Q2: How did you get the data?
A: akshare provides a Python interface to NBS public statistics. I tried 4 sources initially (我的钢铁/兰格钢铁/Kaggle/NBS API); the first three had commercial or data-staleness issues, and the NBS API blocked our IP. akshare was the only viable path.

Q3: How did you ensure no data leakage?
A: I enforced this through code-level assertions at every entry point:
- Train/Val/Test boundary assertions (`len(train)==84`, etc.)
- LSTM grid search accepts only Train data and raises ValueError otherwise
- XGBoost features use `shift(1)` so rolling features only see past values
- LSTM scaler is fit only on the corresponding training data
- XGBoost is isolated in a subprocess to avoid runtime conflicts with PyTorch
- Test set participates in no training, tuning, or weight selection

Q4: Why did you build 4 statistical baselines alongside the ML models?
A: In low-volatility regimes, the Naive baseline already achieves very low MAPE. The baselines let me tell whether ML models actually add value over "predict next = previous value".

Q5: Why is the LSTM not significantly better than XGBoost?
A: On 132 monthly points, the deep-learning model cannot fully leverage its capacity. This is consistent with the sample-size constraint. The causal features I built for XGBoost encode domain knowledge directly, which tree models exploit efficiently.

Q6: Can I look at the code?
A: Yes — the project is public on GitHub at `jinliangyue/civil-engineering-dashboard`. The relevant modules are:
- `src/analyzer/monthly_lstm.py` — XGBoost / Prophet / LSTM helpers
- `src/analyzer/ensemble.py` — validation-weighted ensemble
- `src/evaluation/walk_forward.py` — P0.6 walk-forward validation
- `src/evaluation/metrics.py` — unified MAPE / MAE / RMSE / R²

Q7: What would you change next?
A: Add exogenous macro variables (PMI, CPI, energy prices, FX), and produce prediction intervals instead of point predictions.

---

## Data Acquisition Story (current)

When the interviewer asks "How did you get the data?":

- I tried 我的钢铁网 / 兰格钢铁网 first — both require enterprise authentication
- Tried Kaggle and GitHub datasets — too stale (4+ years old)
- Tried the NBS stats.gov.cn API directly — IP was blocked
- **Final solution**: akshare's `macro_china_ppi()` retrieves the same NBS-published monthly PPI series — 132 real observations from 2015-01 to 2025-12

This story demonstrates:
- Realistic data-acquisition iteration in a domain with commercial restrictions
- Engineering problem-solving (tried multiple paths, found a viable open-data API)
- Honesty about which sources failed and why

---

## Deprecated Background (kept for historical reference only)

> The following describes an **earlier version** of this project that used manually-estimated annual data. That version was superseded in P0.1 (commit `587f9c6`) and the fallback dataset has been removed. Do not present this as current project state.

Earlier iterations of this project (before 2026-09-02) used a 4-industry × 11-year = 44 manually-estimated annual dataset as a placeholder while looking for real monthly data. Those estimates were based on the public PPI index range and were clearly labeled as estimates, but they were not from a verifiable source. Once akshare provided a reliable path to the NBS monthly series, the fallback was removed and the current monthly pipeline took over.

The "为什么放弃月度数据（之前试过 4 个数据源都拿不到）" narrative above reflects that intermediate period. The actual outcome was the opposite: akshare succeeded, monthly data is the current source, and the 44-point annual estimates no longer exist in the project.

---

**This document is maintained alongside `README.md` and `docs/PROJECT_STATUS.md` as the canonical interview talking script for the current project version.**
