"""Single source of truth for all numbers shown in the P0.10 UI.

Every formal (locked) figure here mirrors docs/PROJECT_STATUS.md §4 / §5 and
README.md exactly. They are displayed as static text — never recomputed in
the app, never derived from a live run. Live-demo numbers (Forecast page)
are produced by a per-session retrain and are clearly separated.

Deprecated historical numbers (0.241%, 0.283%, 15%, 44-point manual
estimates, old 2026 fallback forecasts) are intentionally NOT in this file.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Page structure
# --------------------------------------------------------------------------
PAGES = [
    'Overview',
    'Data',
    'Explore',
    'Forecast',
    'Model Evaluation',
    'Robustness',
    'About',
]

APP_TITLE = 'China Industrial PPI Analysis & Forecasting Platform'
SIDEBAR_BRAND = 'PPI Analytics'
SIDEBAR_SUB = 'China industrial PPI · time-series analysis & forecasting'
FOOTER_LINE = (
    'Official NBS monthly PPI · 2015–2025 · 132 observations · '
    'Research versions P0.5 (final test) / P0.6 (walk-forward)'
)

# --------------------------------------------------------------------------
# Data facts (verified against data/raw CSV on 2026-09-03)
# --------------------------------------------------------------------------
DATA_FACTS = {
    'observations': 132,
    'start': '2015-01',
    'end': '2025-12',
    'file': 'data/raw/工业PPI_全国月度_2015-2025.csv',
}

# --------------------------------------------------------------------------
# P0.5 Final Test — locked formal results (PROJECT_STATUS §4)
# Final Test window: 2024-01 ~ 2025-12 (24 months, strict OOS holdout).
# Experiment split (P0.5): Train 2015-01~2021-12 (84) / Validation
# 2022-01~2023-12 (24) / Final Test 2024-01~2025-12 (24).
# --------------------------------------------------------------------------
# (model_label, model_key, MAE, RMSE, MAPE_%, R2)
FINAL_TEST_ROWS = [
    ('Naive',                   'naive',        0.3583, 0.4805, 0.3667,  0.5248),
    ('Seasonal Naive',          'seasonal_naive', 1.2625, 1.7599, 1.2904, -5.3757),
    ('Moving Average (w=3)',    'ma',           0.5903, 0.7432, 0.6040, -0.1371),
    ('SES',                     'ses',          0.5507, 0.6948, 0.5633,  0.0062),
    ('Prophet',                 'prophet',     11.7556, 12.4687, 12.0476, -319.0508),
    ('XGBoost',                 'xgboost',      0.3482, 0.4824, 0.3558,  0.5209),
    ('LSTM',                    'lstm',         0.4288, 0.5224, 0.4387,  0.4381),
    ('Ensemble (validation-weighted)', 'ensemble', 0.3473, 0.4589, 0.3551, 0.5664),
]

FINAL_TEST_KPIS = [
    ('ensemble', 'Ensemble MAPE', '0.3551%', 'validation-weighted · R² 0.5664'),
    ('xgboost', 'XGBoost MAPE', '0.3558%', 'single best model · R² 0.5209'),
    ('lstm', 'LSTM MAPE', '0.4387%', 'PyTorch · R² 0.4381'),
]

# Weight block (README, locked from Validation only — used in About).
ENSEMBLE_WEIGHTS = [
    ('Naive', 0.28189), ('Seasonal Naive', 0.03278), ('MA', 0.15227),
    ('SES', 0.11433), ('Prophet', 0.00947), ('XGBoost', 0.24592),
    ('LSTM', 0.16333),
]

# --------------------------------------------------------------------------
# P0.6 Walk-forward — locked formal results (PROJECT_STATUS §5)
# Folds: F1 train 2015-01~2020-12 (72) → test 2021; F2 train →2021-12 (84)
# → test 2022; F3 train →2022-12 (96) → test 2023. 12 test months per fold.
# --------------------------------------------------------------------------
# (model_label, model_key, mean_MAPE_%, std_MAPE_%)
WALK_FORWARD_MEAN_STD = [
    ('Naive',               'naive',          1.0192, 0.2561),
    ('Seasonal Naive',      'seasonal_naive', 7.9091, 0.8075),
    ('Moving Average (w=3)', 'ma',            1.8170, 0.5036),
    ('SES',                 'ses',            2.4818, 0.7270),
    ('Prophet',             'prophet',       12.1357, 2.1644),
    ('XGBoost',             'xgboost',        1.5958, 0.8992),
    ('LSTM',                'lstm',           1.4087, 0.5024),
]

WF_KPIS = [
    ('naive', 'Naive · mean MAPE', '1.02%', 'across 3 folds'),
    ('lstm', 'LSTM · mean MAPE', '1.41%', 'across 3 folds'),
    ('xgboost', 'XGBoost · mean MAPE', '1.60%', 'across 3 folds'),
]

# Per-fold MAPE (%) for the three lead models — README P0.6 table (2dp
# rounding in that record). Folds F1/F2/F3 = test years 2021/2022/2023.
PER_FOLD_MAPE = {
    'naive':  {'F1 (2021)': 1.35, 'F2 (2022)': 0.98, 'F3 (2023)': 0.72},
    'xgboost': {'F1 (2021)': 2.86, 'F2 (2022)': 1.03, 'F3 (2023)': 0.89},
    'lstm':   {'F1 (2021)': 1.56, 'F2 (2022)': 1.94, 'F3 (2023)': 0.73},
}
PER_FOLD_YEARS = ['F1 (2021)', 'F2 (2022)', 'F3 (2023)']
PER_FOLD_MODELS = ['naive', 'xgboost', 'lstm']  # only these were recorded per fold

FOLDS = [
    ('F1', 'Train 2015-01 – 2020-12 (72)', 'Test 2021-01 – 2021-12 (12)'),
    ('F2', 'Train 2015-01 – 2021-12 (84)', 'Test 2022-01 – 2022-12 (12)'),
    ('F3', 'Train 2015-01 – 2022-12 (96)', 'Test 2023-01 – 2023-12 (12)'),
]

# --------------------------------------------------------------------------
# Model catalog (About page)
# --------------------------------------------------------------------------
MODEL_CATALOG = [
    ('Naive', 'Baseline', 'y[t+1] = y[t]'),
    ('Seasonal Naive', 'Baseline', 'y[t+1] = y[t-12]'),
    ('Moving Average', 'Baseline', 'y[t+1] = mean(y[t-3:t]), window=3'),
    ('SES', 'Baseline', 'Simple exponential smoothing, alpha=0.3'),
    ('Prophet', 'Statistical ML', 'Additive trend + yearly seasonality'),
    ('XGBoost', 'Gradient boosting', '15 hand-crafted causal features (lag / rolling / YoY / MoM)'),
    ('LSTM', 'Deep learning', '2-layer PyTorch LSTM, rolling one-step-ahead'),
]

PIPELINE_STEPS = [
    'Official data',
    'Data validation',
    'Time-based split',
    'Model training',
    'Validation (weights)',
    'Final Test',
    'Walk-forward validation',
]

# --------------------------------------------------------------------------
# Copy: badges, caveats, captions (exact strings shown in the UI)
# --------------------------------------------------------------------------
BADGE_LOCKED_RESEARCH = 'Locked Research Result'
BADGE_FORMAL = 'Formal Evaluation'
BADGE_LIVE_DEMO = 'Live Demo'
BADGE_INTERACTIVE = 'Interactive Demo'

DEMO_DISCLAIMER = (
    'Models are retrained for interactive demonstration. '
    'Results may vary slightly depending on runtime environment.'
)

REGIME_CAVEAT = (
    'The 2024–2025 test period was relatively low-volatility '
    '(annual PPI range ≈ 2.1 in 2024, ≈ 1.7 in 2025). Results should not be '
    'generalized beyond this regime.'
)

FINAL_TEST_SUBTITLE = (
    'Strict out-of-sample holdout, 2024-01 – 2025-12 (24 months). '
    'Locked research results — not recomputed in this app.'
)

# Evaluation page explanatory box
EVAL_INFO = (
    'Read these numbers together with the walk-forward results on the '
    'Robustness page. In this low-volatility window even the Naive baseline '
    'reaches 0.37% MAPE, so small MAPE values reflect the regime as well as '
    'model skill — that is why the formal framing does not claim a universal '
    '"accuracy" figure.'
)

# Forecast page
FORECAST_HEADING_NOTE = (
    'Three models are retrained on the official 132 observations inside this '
    'session, then asked for a 2026 out-of-sample demo forecast. This is an '
    'interactive demonstration — not a locked research result.'
)

WF_NO_ENSEMBLE_NOTE = (
    'Walk-forward Ensemble is intentionally not reported: P0.5 ensemble '
    'weights were locked from one fixed Validation period, and recomputing '
    'weights per fold would be a structurally different experiment.'
)

FOLD_NOISE_NOTE = (
    'Each fold tests on only 12 months, so individual fold MAPE values carry '
    'meaningful noise; the mean across folds is the more stable comparison.'
)

FOOTER_QUOTES = [
    'Performance varies across historical regimes. No single model dominates '
    'across all folds.',
    'Each fold uses an expanding training window and a subsequent '
    'out-of-sample test period.',
]

DATA_CHECKLIST = [
    ('Official source', 'National Bureau of Statistics monthly PPI release; initially fetched via akshare, committed to the repo.'),
    ('Real monthly observations', '132 points, 2015-01 – 2025-12 — one per month, no gaps.'),
    ('No manually estimated fallback data', 'A previous version used 44 manually-estimated annual points; they were removed and are not part of this platform.'),
    ('Index semantics', 'Index of the prior-year same month = 100; YoY and YTD fields come from the same release.'),
    ('Offline at runtime', 'The app reads the committed CSV; akshare is only the historical fetch tool, no network call at runtime.'),
    ('Schema validation before render', 'The app checks file presence and required columns, and stops with a clear error otherwise.'),
]

# --------------------------------------------------------------------------
# Data columns
# --------------------------------------------------------------------------
DATA_COLUMNS = [
    ('date', 'Month (first day of month)'),
    ('ppi_index', 'PPI index, prior-year same month = 100'),
    ('yoy_pct', 'Year-over-year change, % (release value)'),
    ('ytd_index', 'Year-to-date index'),
]

# --------------------------------------------------------------------------
# About page copy
# --------------------------------------------------------------------------
ABOUT_INTRO = (
    'A time-series analysis and forecasting platform for China’s Producer '
    'Price Index (PPI): statistical baselines, Prophet, XGBoost, a PyTorch '
    'LSTM, and a validation-weighted ensemble — all evaluated under strict '
    'out-of-sample protocols on 132 real monthly observations from the '
    'National Bureau of Statistics (2015-01 to 2025-12).'
)

REPRO_COMMANDS = [
    ('Load the 132 monthly PPI points', "python3 -c \"from src.ppi_monthly import load_monthly_ppi; print(load_monthly_ppi().shape)\""),
    ('Run unified metrics unit tests', 'python3 -m src.evaluation.test_metrics'),
    ('Reproduce P0.5 (final test)', 'python3 -m src.analyzer.ensemble'),
    ('Reproduce P0.6 (walk-forward)', 'python3 -m src.evaluation.walk_forward'),
    ('Print environment fingerprint', 'python3 scripts/environment_fingerprint.py'),
]

LIMITATIONS = [
    'Sample size: 132 monthly observations total; each walk-forward fold has only 12 test points, so fold-level MAPE carries noise.',
    'Single time series: no exogenous variables (macro indicators, energy prices, FX, commodity prices) are included.',
    'Low-volatility regime caveat: the 2024–2025 final-test window is a calm PPI regime; low MAPE numbers must be read alongside the Naive baseline.',
    'No walk-forward ensemble: weights were locked once from the Validation period (see Robustness page).',
    'No uncertainty quantification: point predictions and point metrics only — no prediction intervals.',
    'Demonstration, not deployment: results demonstrate a pipeline and evaluation methodology, not a guarantee of future PPI values.',
]

GITHUB_URL = 'https://github.com/jinliangyue/civil-engineering-dashboard'
APP_URL = 'https://civil-engineering-ppi.streamlit.app/'
