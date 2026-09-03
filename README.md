# China Industrial PPI Time-Series Analysis and Forecasting Platform

A time-series analysis and forecasting platform for China's Producer Price Index (PPI), combining statistical baselines, machine learning, deep learning, and validation-weighted ensemble methods under strict out-of-sample (OOS) evaluation.

Author: jinliangyue (pen-name: 十八 / "Eighteen")
Goal: 2026 autumn recruiting portfolio project
Live demo: https://civil-engineering-ppi.streamlit.app/
GitHub: jinliangyue/civil-engineering-dashboard

---

## Overview

This project builds an end-to-end monthly PPI forecasting pipeline using 132 real monthly PPI observations (2015-01 to 2025-12) sourced from China's National Bureau of Statistics via the akshare library.

The platform evaluates 7 models (4 statistical baselines + Prophet + XGBoost + LSTM) under two complementary evaluation regimes:

- **Final Test (P0.5)**: strict OOS 24-month holdout (2024-01 ~ 2025-12)
- **Walk-forward Validation (P0.6)**: 3 expanding-window folds over historical years (F1=2021, F2=2022, F3=2023)

All experimental boundaries (Train/Validation/Test), leakage controls (causal features, scaler isolation, rolling one-step-ahead prediction), and hyperparameter isolation are enforced by code-level assertions.

---

## Why This Project

China's PPI is the key benchmark index for material price adjustment ("材料调差公式") in engineering cost contracts. Engineering cost estimators need both an understanding of historical price dynamics and a defensible forecast for upcoming project bids.

Public PPI data is available but scattered across monthly statistical bulletins without cross-industry comparison tools. This project demonstrates:

1. End-to-end data acquisition from a public source (akshare → NBS)
2. Rigorous time-series evaluation under strict OOS constraints
3. Comparison of simple baselines against ML/DL models on real economic data
4. Validation-weighted ensemble construction
5. Walk-forward robustness validation across multiple historical windows

---

## Data

| Field | Value |
|---|---|
| Source | akshare `macro_china_ppi()` → National Bureau of Statistics monthly PPI release |
| File | `data/raw/工业PPI_全国月度_2015-2025.csv` |
| Frequency | Monthly |
| Time range | 2015-01 ~ 2025-12 |
| Observations | 132 |
| Index meaning | PPI index, prior-year same month = 100 |

**Historical fallback data removed**: A previous version of this project used 44 manually-estimated annual observations for 4 industries. Those have been removed because they were not from a verifiable source and would have introduced leakage and unverifiable claims. The current project uses only the 132-point monthly series above.

---

## Methodology

### Models evaluated (7)

| # | Model | Type | Implementation |
|:--:|---|---|---|
| 1 | Naive | Baseline | y[t+1] = y[t] |
| 2 | Seasonal Naive | Baseline | y[t+1] = y[t-12] |
| 3 | Moving Average | Baseline | y[t+1] = mean(y[t-3:t]) (window=3) |
| 4 | SES | Baseline | Simple exponential smoothing (alpha=0.3) |
| 5 | Prophet | Statistical ML | Additive trend + yearly seasonality |
| 6 | XGBoost | Gradient boosting | 15 hand-crafted features (lag/rolling/yoy/mom), all causal |
| 7 | LSTM (PyTorch) | Deep learning | 2-layer LSTM with rolling one-step-ahead prediction |

### Validation-weighted Ensemble

Ensemble weights are derived exclusively from the Validation period and locked before Final Test evaluation:

```
raw_weight_i = 1 / Validation_MAPE_i
weight_i = raw_weight_i / sum(raw_weight)
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

Test actuals are not used in weight computation.

---

## Experimental Design

### Train / Validation / Test split (P0.5)

| Segment | Time range | Points |
|---|---|---:|
| Train | 2015-01 ~ 2021-12 | 84 |
| Validation | 2022-01 ~ 2023-12 | 24 |
| Final Test | 2024-01 ~ 2025-12 | 24 |
| **Total** | 2015-01 ~ 2025-12 | **132** |

### Walk-forward Validation (P0.6)

| Fold | Train | Test |
|---|---|---|
| F1 | 2015-01 ~ 2020-12 (72) | 2021-01 ~ 2021-12 (12) |
| F2 | 2015-01 ~ 2021-12 (84) | 2022-01 ~ 2022-12 (12) |
| F3 | 2015-01 ~ 2022-12 (96) | 2023-01 ~ 2023-12 (12) |

### Leakage Control

The implementation includes explicit leakage controls:

- **Data split assertion**: `len(train)=84, len(val)=24, len(test)=24` enforced at every entry point.
- **Hyperparameter isolation**: LSTM grid search uses only the Train segment (84 points).
- **Scaler isolation**: LSTM `StandardScaler.fit()` is called only on the corresponding training data per fold.
- **Causal features**: XGBoost lag / rolling / YoY / MoM features use only information available before each prediction time (`shift(1)` applied before rolling).
- **Rolling prediction**: XGBoost and LSTM use one-step-ahead rolling prediction, appending each test actual to history only after that point is predicted.
- **Ensemble**: Weights derived from Validation only; locked before Test evaluation.
- **Test isolation**: Test set participates in no training, no tuning, no scaler fitting, and no weight calculation.

The phrase "zero leakage" is intentionally avoided. What is asserted is the explicit set of checks above.

---

## Results

### Final Test (P0.5) — 2024-01 to 2025-12, 24 months OOS

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Naive | 0.3583 | 0.4805 | 0.3667% | 0.5248 |
| MA | 0.5903 | 0.7432 | 0.6040% | -0.1371 |
| SES | 0.5507 | 0.6948 | 0.5633% | 0.0062 |
| XGBoost | 0.3482 | 0.4824 | **0.3558%** | 0.5209 |
| LSTM | 0.4288 | 0.5224 | **0.4387%** | 0.4381 |
| **Ensemble (validation-weighted)** | **0.3473** | **0.4589** | **0.3551%** | **0.5664** |

*Note: 2024-2025 is a low-volatility PPI regime (annual range ≈ 2.1 in 2024, ≈ 1.7 in 2025). The Naive baseline already achieves MAPE ≈ 0.37%. The low ensemble MAPE reflects this regime, not a universal claim of forecasting power.*

### Walk-forward Validation (P0.6)

| Model | F1 (2021) | F2 (2022) | F3 (2023) | Mean | Std |
|---|---:|---:|---:|---:|---:|
| Naive | 1.35% | 0.98% | 0.72% | **1.02%** | 0.26% |
| MA | 2.29% | 2.04% | 1.12% | 1.82% | 0.50% |
| XGBoost | 2.86% | 1.03% | 0.89% | **1.60%** | 0.90% |
| LSTM | 1.56% | 1.94% | 0.73% | **1.41%** | 0.50% |

*Walk-forward Ensemble was not recomputed. P0.5 ensemble weights were specifically derived from a fixed Validation period. Recomputing weights inside each historical fold would constitute a structurally different experiment.*

### Why P0.5 and P0.6 results differ

P0.5 evaluates 2024-2025, a low-volatility PPI regime (annual range ≈ 1.7-2.1). The Naive baseline itself achieves MAPE ≈ 0.37%. P0.6 covers 2021 (high), 2022 (transition), 2023 (declining), where variance is much larger (annual range up to 13.2 in 2021). P0.5 and P0.6 measure different things and cannot be directly compared by magnitude alone.

---

## Key Findings

1. The Naive baseline is competitive in low-volatility regimes, confirming that low MAPE numbers on stable periods do not by themselves demonstrate model skill.
2. LSTM underperforms XGBoost in both regimes on this dataset. This is consistent with the small-sample constraint (132 points, of which the LSTM uses ~84 for training) and the strong causal feature engineering XGBoost receives.
3. Prophet performs poorly (R² strongly negative) on both Test and Walk-forward. Its trend component extrapolates training-period upward trends that do not match the actual PPI trajectory.
4. Validation-weighted ensemble achieves the highest R² on Test (0.5664) but only marginally improves over the single best MAPE (0.3551% vs XGBoost 0.3558%). The ensemble's strength here is robustness, not headline accuracy.
5. Across 3 Walk-forward folds, Naive, XGBoost, and LSTM are all within ~0.5% MAPE of each other on average. None of the models consistently dominates.

---

## Engineering Implementation

### XGBoost subprocess isolation

XGBoost training and prediction are isolated in a separate Python subprocess (`spawn` start method, hard 60-second timeout, terminate on timeout). This isolation:

- Ensures XGBoost's C++ runtime resources are released before PyTorch LSTM runs in the main process.
- Allows the LSTM to run cleanly after XGBoost completes in the parent process.
- Avoids runtime-level state conflicts observed in the same-process alternative.

This is documented in the codebase as a deliberate engineering decision, not a workaround for a generic library defect.

### Unified evaluation metrics

All MAPE, MAE, RMSE, and R² values across the project are computed via a single shared module (`src/evaluation/metrics.py`) with consistent semantics:

- MAPE: filters zero-y_true entries; raises on all-zero y_true.
- R²: raises on constant y_true (consistent with `sklearn.metrics.r2_score`).
- Inputs accept list / numpy / pandas Series uniformly.

---

## Tech Stack

- **Research environment**: Python 3.9.13 (locked — see `docs/ENVIRONMENT.md`, `requirements-research.txt`)
- **Data**: pandas, numpy
- **Statistics / ML**: scipy, Prophet, XGBoost, scikit-learn
- **Deep learning**: PyTorch (LSTM)
- **Data acquisition**: akshare (NBS PPI monthly)
- **Web app**: Streamlit + Streamlit Cloud (platform-managed Python runtime; `runtime.txt` = `python-3.12` is repository configuration and has not been independently verified against the live Cloud runtime)
- **Visualization**: Plotly

---

## Project Structure

```
civil-engineering-dashboard/
├── README.md                          (this file)
├── requirements.txt                   Legacy unpinned install list (what Cloud installs today)
├── requirements-research.txt          Locked research env (Python 3.9.13 matrix)
├── requirements-deploy.txt            Deployment compatibility baseline (Cloud not verified)
├── runtime.txt                        Python version pin (3.12, repository config only)
├── app/
│   └── streamlit_app.py               Streamlit dashboard
├── data/
│   └── raw/工业PPI_全国月度_2015-2025.csv
├── docs/
│   ├── PROJECT_STATUS.md              Full project state (Chinese)
│   ├── ENVIRONMENT.md                 Environment lock & research-vs-demo separation
│   ├── interview_script_ml.md        Interview talking points
│   ├── DEPLOYMENT.md                  Deployment guide
│   ├── WORKFLOW.md
│   ├── DATA_INPUT_SPEC.md
│   ├── data_sources.md
│   ├── resume_description.md         Resume-ready bullets
│   └── LSTM_TUNING_RESULTS.md        Historical LSTM tuning notes
├── scripts/
│   ├── environment_fingerprint.py     Env fingerprint (versions + Prophet MD5)
│   ├── fetch_ppi.py                  Fetch PPI via akshare
│   └── run_pipeline.py
└── src/
    ├── ppi_monthly.py                Monthly data loader
    ├── analyzer/
    │   ├── monthly_lstm.py            LSTM + Prophet + XGBoost helpers
    │   ├── lstm_tuning.py             Grid search (Train only)
    │   └── ensemble.py                Validation-weighted ensemble
    └── evaluation/
        ├── metrics.py                 MAPE / MAE / RMSE / R²
        ├── test_metrics.py            metrics unit tests
        └── walk_forward.py            P0.6 walk-forward validation
```

---

## Reproducibility

```bash
# Install dependencies
pip3 install -r requirements.txt

# Load 132 monthly PPI points
python3 -c "from src.ppi_monthly import load_monthly_ppi; print(load_monthly_ppi().shape)"

# Run unified metrics unit tests
python3 -m src.evaluation.test_metrics

# Run P0.5 reproduction (validation-weighted ensemble on Final Test 2024-01 ~ 2025-12)
python3 -m src.analyzer.ensemble

# Run P0.6 reproduction (Walk-forward Validation on 2021/2022/2023)
python3 -m src.evaluation.walk_forward

# Print the environment fingerprint (versions + Prophet binary MD5)
python3 scripts/environment_fingerprint.py
```

### Environment lock and research vs live demo

The formal research results above (0.3551% etc.) are **locked to the verified research environment** (Python 3.9.13 matrix; see `docs/ENVIRONMENT.md` and `requirements-research.txt`). They are 2024-2025 low-volatility-regime results, not a guarantee that every run reproduces them.

The Streamlit demo (`app/streamlit_app.py`) **re-trains three models live on every session** (Prophet / XGBoost / LSTM), using the same 108/24 split as the formal experiment and the P0.3-locked LSTM hyperparameters (P0.9.6 wiring; no per-session grid search, no per-session ensemble re-run — the locked 7-model ensemble results live on Tab 5 only). The demo metrics are **not** the formal research results — they may drift across environments and runs. Do not quote demo numbers as the locked research result.

The streamlit app runs at https://civil-engineering-ppi.streamlit.app/. Its runtime is managed by Streamlit Cloud; deployment dependencies are documented in `requirements-deploy.txt` (cloud runtime itself has not been independently verified).

---

## Limitations

1. **Sample size**: Only 132 monthly observations total. Walk-forward folds contain only 12 test points each, so individual fold MAPE values carry meaningful noise.
2. **Single time series**: The current model uses only the PPI index itself as input. No exogenous variables (macro indicators, energy prices, FX, commodity prices) are included.
3. **Low-volatility regime caveat**: The 2024-2025 Final Test window is a low-volatility PPI regime (annual range 1.7-2.1). Low MAPE numbers in this window should be interpreted alongside the Naive baseline, not as a universal claim of forecasting accuracy.
4. **Walk-forward ensemble**: Ensemble weights were derived once from the Validation period and locked. They were not recomputed for each Walk-forward fold, so Walk-forward Ensemble results are not reported.
5. **No uncertainty quantification**: This project reports point predictions and point metrics only. No prediction intervals or probabilistic forecasts.
6. **Demonstration, not deployment**: The results demonstrate a forecasting pipeline and evaluation methodology. They are not intended as guarantees of future PPI values.

---

## Future Work

- Incorporate exogenous macro variables (PMI, CPI, energy prices, FX)
- Multivariate time-series models
- Longer time-series horizons with quarterly / annual aggregations
- Stricter rolling-origin evaluation
- Prediction interval / uncertainty estimation
- 2026 future-forecast monitoring (only after the validation pipeline above is fully reproducible)

---

## License

MIT (for portfolio / educational use).
