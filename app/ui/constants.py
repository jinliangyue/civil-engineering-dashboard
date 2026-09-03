"""Single source of truth for all numbers and display copy — P0.11 UI.

Every formal (locked) figure here mirrors docs/PROJECT_STATUS.md §4 / §5 and
README.md exactly. They are displayed as static text — never recomputed in
the app, never derived from a live run. Live-demo numbers (Forecast page)
are produced by a per-session retrain and are clearly separated.

Numbers are frozen: the redesign (P0.11) is a pure display-layer change and
must not alter any locked figure. Model names stay in English throughout
(they are model identifiers); page copy is Chinese-first.

Deprecated historical numbers (0.241%, 0.283%, 15%, 44-point manual
estimates, old 2026 fallback forecasts) are intentionally NOT in this file —
they appear only as an honesty note on the Methodology page.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Page structure
# --------------------------------------------------------------------------
PAGES = [
    '总览',
    '数据',
    '趋势分析',
    '预测',
    '模型评估',
    '稳健性检验',
    '方法与说明',
]

APP_TITLE = '工业 PPI 分析与预测平台 · China Industrial PPI Analytics'
SIDEBAR_BRAND = '工业 PPI'
SIDEBAR_SUB = 'Research analytics · terminal'
FOOTER_LINE = (
    '国家统计局官方月度 PPI · 2015–2025 · 132 个观测 · '
    '研究版本 P0.5（最终测试）/ P0.6（滚动外推）'
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
    ('ensemble', 'Ensemble', '0.3551%', '验证加权 · R² 0.5664'),
    ('xgboost', 'XGBoost', '0.3558%', '最优单一模型 · R² 0.5209'),
    ('lstm', 'LSTM', '0.4387%', 'PyTorch · R² 0.4381'),
]

# Weight block (README, locked from Validation only — Methodology page).
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
    ('naive', 'Naive', '1.02%', '3 折平均 MAPE % · ± 0.26'),
    ('lstm', 'LSTM', '1.41%', '3 折平均 MAPE % · ± 0.50'),
    ('xgboost', 'XGBoost', '1.60%', '3 折平均 MAPE % · ± 0.90'),
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
    ('F1', '训练窗口 2015-01 – 2020-12（72 个月）', '测试 2021 年（12 个月）'),
    ('F2', '训练窗口 2015-01 – 2021-12（84 个月）', '测试 2022 年（12 个月）'),
    ('F3', '训练窗口 2015-01 – 2022-12（96 个月）', '测试 2023 年（12 个月）'),
]

# --------------------------------------------------------------------------
# Model catalog (Methodology page)
# --------------------------------------------------------------------------
MODEL_CATALOG = [
    ('Naive', '基线模型', 'y[t+1] = y[t]'),
    ('Seasonal Naive', '基线模型', 'y[t+1] = y[t-12]'),
    ('Moving Average', '基线模型', 'y[t+1] = mean(y[t-3:t])，窗口 3'),
    ('SES', '基线模型', '简单指数平滑，alpha=0.3'),
    ('Prophet', '统计机器学习', '加性趋势 + 年度季节性'),
    ('XGBoost', '梯度提升', '15 个手工因果特征（滞后 / 滚动 / 同比 / 环比）'),
    ('LSTM', '深度学习', '2 层 PyTorch LSTM，滚动单步外推'),
]

# (cn, en) — rendered as numbered pipeline steps on the Methodology page.
PIPELINE_STEPS = [
    ('官方数据', 'Official data'),
    ('数据校验', 'Data validation'),
    ('时间切分', 'Time-based split'),
    ('模型训练', 'Model training'),
    ('验证加权', 'Validation (weights)'),
    ('最终测试', 'Final Test'),
    ('滚动外推诊断', 'Walk-forward diagnostic'),
]

# --------------------------------------------------------------------------
# Copy: badges, caveats, captions (exact strings shown in the UI)
# --------------------------------------------------------------------------
BADGE_LOCKED_RESEARCH = '正式研究结果 · LOCKED'
BADGE_FORMAL = '正式评估 · FORMAL'
BADGE_LIVE_DEMO = '现场演示 · LIVE DEMO'
BADGE_INTERACTIVE = '交互演示 · DEMO'

DEMO_DISCLAIMER = (
    '以下模型在本会话内重新训练，仅用于交互演示。'
    '结果会随运行环境略有浮动，不与正式锁定数字混用。'
)

REGIME_CAVEAT = (
    '2024–2025 最终测试窗口处于低波动行情区间'
    '（PPI 年度振幅 2024 年约 2.1、2025 年约 1.7）。'
    '该数字不应外推到此区间之外的行情。'
)

FINAL_TEST_SUBTITLE = (
    '严格的样本外留出测试：2024-01 – 2025-12（24 个月）。'
    '锁定研究结果，不在本应用中重算。'
)

# Model Evaluation page explanatory block
EVAL_INFO = (
    '请与「稳健性检验」页的滚动外推结果对照阅读。这一低波动窗口内'
    '连 Naive 基线都达到 0.37% 的 MAPE，低数字同时反映行情区间与模型'
    '能力，因此本平台不使用绝对的准确率式表述。'
)

# Forecast page
FORECAST_HEADING_NOTE = (
    '三个模型在本会话内基于官方 132 条观测重新训练，'
    '随后外推 2026 全年（12 个月）作为样本外演示预测。'
    '这是交互演示，不是锁定研究结果。'
)

WF_NO_ENSEMBLE_NOTE = (
    '滚动外推刻意不报告 Ensemble：P0.5 集成权重只在单一验证窗口锁定'
    '一次；若每折重算权重，属于结构不同的另一组实验。'
)

FOLD_NOISE_NOTE = (
    '每折只覆盖 12 个测试月，单折 MAPE 携带的噪声较大；'
    '跨折均值是比较模型更稳的指标。'
)

FOOTER_QUOTES = [
    '不同历史行情下模型表现不一，没有单一模型在所有折中持续占优。',
    '每折使用扩张式训练窗口，外推随后的样本外年份。',
]

DATA_CHECKLIST = [
    ('官方来源', '国家统计局月度 PPI 发布数据，经 akshare 一次性获取后提交入库。'),
    ('真实月度观测', '132 个点，2015-01 – 2025-12，逐月无缺口。'),
    ('不含人工估算兜底', '旧版 44 个年度人工估算点已移除，不属于本平台数据。'),
    ('指数口径', '指数以上年同月为 100；同比与累计字段出自同一次发布。'),
    ('运行时不联网', '应用只读已提交的 CSV；akshare 仅用于历史获取，页面加载不发网络请求。'),
    ('渲染前校验', '应用先检查文件存在与必需列完整，不满足时报错并停止。'),
]

# --------------------------------------------------------------------------
# Data columns
# --------------------------------------------------------------------------
DATA_COLUMNS = [
    ('date', '月份（每月 1 日）'),
    ('ppi_index', 'PPI 指数（上年同月 = 100）'),
    ('yoy_pct', '同比变化 %（发布值）'),
    ('ytd_index', '累计指数'),
]

# --------------------------------------------------------------------------
# Methodology page copy
# --------------------------------------------------------------------------
METHOD_INTRO = (
    '面向中国工业生产者出厂价格指数（PPI）的时序分析与预测平台：'
    '统计基线、Prophet、XGBoost、PyTorch LSTM 与验证加权集成，'
    '全部在国家统计局 132 条真实月度观测（2015-01 – 2025-12）上'
    '按严格样本外协议评估。'
)

REPRO_COMMANDS = [
    ('加载 132 条月度 PPI', "python3 -c \"from src.ppi_monthly import load_monthly_ppi; print(load_monthly_ppi().shape)\""),
    ('运行统一指标单元测试', 'python3 -m src.evaluation.test_metrics'),
    ('复现 P0.5（最终测试）', 'python3 -m src.analyzer.ensemble'),
    ('复现 P0.6（滚动外推）', 'python3 -m src.evaluation.walk_forward'),
    ('打印环境指纹', 'python3 scripts/environment_fingerprint.py'),
]

LIMITATIONS = [
    '样本量：共 132 条月度观测；滚动外推每折只有 12 个测试点，单折 MAPE 带噪声。',
    '单一序列：不含外生变量（宏观指标、能源价格、汇率、大宗商品价格等均未纳入）。',
    '低波动区间提醒：2024–2025 最终测试窗口属平稳行情，低 MAPE 必须与 Naive 基线对照阅读。',
    '无滚动外推集成：权重只在验证期锁定一次（见「稳健性检验」页）。',
    '无不确定性量化：只有点预测与点指标，没有预测区间。',
    '演示而非投产：本平台演示的是流程与评估方法，不构成对未来 PPI 数值的保证。',
]

# Honesty note — deprecated history, Methodology page only.
DEPRECATED_NOTE = (
    '历史说明：旧版曾使用 4 个行业 × 11 年共 44 个年度人工估算点，'
    '并据此报告过 0.241% / 0.283% 及 2026 年 98.9 / 106.5 / 110.4 / 116.0 '
    '的年度预测。那些数字基于占位估算，已在 P0.9 随估算数据一并移除，'
    '与本平台当前展示的 132 条官方月度观测无关。'
)

GITHUB_URL = 'https://github.com/jinliangyue/civil-engineering-dashboard'
APP_URL = 'https://civil-engineering-ppi.streamlit.app/'
