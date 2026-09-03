"""预测 Forecast — live interactive demo, clearly separated from formal results.

Three models are retrained per session on the official 132 observations
(108-point training window through 2023-12, same split as the formal
experiment), then two things are produced from the same fitted models:

1. Future demo forecast 2026-01 -> 2026-12 (no ground truth yet exists:
   no MAPE is shown, this is explicitly not a test result).
2. Out-of-sample backtest demo 2024-01 -> 2025-12 (real values exist, so
   demo MAPE is shown and labeled as a live re-run, not the locked result).

Model wiring (no research code is modified — only called):
- Prophet:  train_prophet_final returns the fitted model; the future helper
  counts 36 periods from the model's own fit end (2023-12), so the app
  filters to date >= 2026-01-01.
- XGBoost:  predict_xgboost_future_recursive is seeded from the FULL series
  (132 real values through 2025-12) -> first forecast lands on 2026-01.
- LSTM:     predict_lstm_future_recursive likewise seeds its last window
  from real values through 2025-12; the scaler mean/std were fitted on the
  108 training points only (P0.3-locked hyperparameters).

P0.11: display layer only. Payload / training logic below is unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C

_PAYLOAD_KEY = 'p010_demo_payload'
_TRAIN_SPLIT_END = pd.Timestamp('2023-12-31')
_TEST_START = pd.Timestamp('2024-01-01')
_FUTURE_START = pd.Timestamp('2026-01-01')

MODEL_LABELS = {
    'prophet': 'Prophet',
    'xgboost': 'XGBoost',
    'lstm': 'LSTM',
}

_ALL_KEY = 'all'
_SHOW_OPTIONS = [
    (_ALL_KEY, '全部三个模型（All three）'),
    ('prophet', 'Prophet'),
    ('xgboost', 'XGBoost'),
    ('lstm', 'LSTM'),
]


def render(df: pd.DataFrame) -> None:
    # --- Page head --------------------------------------------------------
    X.page_head(
        'FORECAST · 预测演示',
        '预测',
        subtitle=(
            '交互演示：三个模型在本会话内重训并外推 2026 全年。'
            '这里的数字是现场演示，不是锁定研究结果。'
        ),
        badges=[X.demo_badge(f'{C.BADGE_LIVE_DEMO} · {C.BADGE_INTERACTIVE}')],
    )

    payload = _get_payload(df)
    if payload is None:
        return  # error already shown

    # --- LIVE DEMO stage --------------------------------------------------
    X.stage_head(
        'demo',
        '现场演示 · 2026 年交互外推',
        badge_html=X.demo_badge(),
    )
    X.note(C.FORECAST_HEADING_NOTE)
    X.demo_disclaimer_note()

    which = st.radio(
        '图表显示模型',
        options=[label for _, label in _SHOW_OPTIONS],
        index=0,
        horizontal=True,
        key='p10_forecast_show',
    )
    key_sel = next(k for k, v in _SHOW_OPTIONS if v == which)
    selected = ['prophet', 'xgboost', 'lstm'] if key_sel == _ALL_KEY \
        else [key_sel]

    _future_chart(df, payload, selected)
    X.block(
        '外推区间 2026-01 → 2026-12：2025-12 之后尚不存在真实值，'
        '因此本图不提供任何 MAPE——这不是测试结果。'
        '这些线条演示的是完整流水线，不是对未来 PPI 数值的承诺。',
        kind='demo',
    )

    # --- Same-session OOS backtest sub-block -------------------------------
    X.section_head(
        kicker='BACKTEST · 同期回测',
        title='样本外回测（2024-01 → 2025-12 · 同一会话同一批模型）',
        subtitle=(
            '在真实值已存在的 24 个月上做滚动单步外推，因此可以给出演示 '
            'MAPE。这些数字是本次会话的现场重跑；锁定正式结果见'
            '「模型评估」页。'
        ),
    )
    _backtest_chart(payload)
    X.show_dataframe(_demo_metrics_frame(payload), height=210)
    X.note_sm(
        '演示指标（MAE / RMSE / MAPE / R²）由研究流水线同款统一指标模块'
        '计算，但来自本会话的全新重训——数值与锁定结果之间会有环境漂移'
        '（numbers drift across environments）。锁定正式值：'
        'Ensemble 0.3551% · XGBoost 0.3558% · '
        'Naive 0.3667% · LSTM 0.4387% MAPE（最终测试）。'
    )

    # --- FORMAL (locked) stage ----------------------------------------------
    X.stage_head(
        'formal',
        '正式研究结果 · 2024–2025 最终测试',
        badge_html=X.locked_badge(C.BADGE_LOCKED_RESEARCH),
    )
    X.note(
        '正式数字在研究环境（Python 3.9.13 锁定矩阵）中产出并锁定，'
        '不在本页重新运行；上方演示只用于交互展示。'
    )
    X.kpi_strip([
        ('Ensemble', '0.3551%', '验证加权 · 正式锁定', 'accent'),
        ('XGBoost', '0.3558%', '最优单一模型', 'accent'),
        ('Naive', '0.3667%', '基线对照'),
        ('LSTM', '0.4387%', 'PyTorch'),
    ], variant='lean')
    X.jump_button('查看完整 8 模型锁定表 →', '模型评估')


# --------------------------------------------------------------------------
# Payload: one per-session training run, memoized in st.session_state
# --------------------------------------------------------------------------
def _get_payload(df: pd.DataFrame) -> Optional[Dict]:
    if _PAYLOAD_KEY not in st.session_state:
        try:
            with st.spinner(
                'Training Prophet + XGBoost + LSTM on the 132 official '
                'observations — one-time per session (~30–90 s)…'
                '（本会话仅训练一次）'
            ):
                st.session_state[_PAYLOAD_KEY] = _train_payload(df)
        except Exception as exc:  # noqa: BLE001 — show honest error, keep app alive
            st.session_state.pop(_PAYLOAD_KEY, None)
            st.error(
                f'Session training failed ({type(exc).__name__}). '
                f'The page is left without demo numbers rather than shown '
                f'invented ones. Please reload the app and retry. '
                f'（会话内训练失败：本页宁可不显示演示数字，'
                f'也不显示虚构数字。请刷新应用后重试。）'
            )
            return None
    return st.session_state[_PAYLOAD_KEY]


def _train_payload(df: pd.DataFrame) -> Dict:
    # Import order matters: heavy libs are imported only here (first visit of
    # this page), and torch thread limits are set before any training so the
    # OpenMP runtime cannot deadlock (libomp conflict seen without guards).
    from src.analyzer.monthly_lstm import (
        get_monthly_feature_columns,
        predict_lstm_future_recursive,
        predict_prophet_future,
        predict_xgboost_future_recursive,
        train_lstm_final,
        train_prophet_final,
        train_xgboost_final,
    )

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already initialized — fine, limits already set
        pass

    df = df.sort_values('date').reset_index(drop=True)
    train_val = df[df['date'] <= _TRAIN_SPLIT_END].reset_index(drop=True)
    test = df[df['date'] >= _TEST_START].reset_index(drop=True)
    assert len(train_val) == 108, f'train+val must be 108, got {len(train_val)}'
    assert len(test) == 24, f'test must be 24, got {len(test)}'

    from src.analyzer.ensemble import LSTM_BEST_PARAMS

    p_metrics, p_pred, p_model = train_prophet_final(train_val, test)
    x_metrics, x_preds, x_model, _ = train_xgboost_final(train_val, test)
    l_metrics, l_preds, l_model, (mean, std), l_actuals = train_lstm_final(
        train_val, test, LSTM_BEST_PARAMS
    )

    # Future demo forecasts. XGBoost/LSTM helpers seed from whatever df they
    # receive: pass the FULL series so recursion starts from 2025-12 real
    # values and the first step lands on 2026-01. Prophet's helper counts
    # from its own fit end (2023-12): request 36, filter to 2026 onward.
    future = {
        'prophet': [
            r for r in predict_prophet_future(p_model, horizon=36)
            if r['date'] >= _FUTURE_START
        ],
        'xgboost': predict_xgboost_future_recursive(
            x_model, df, get_monthly_feature_columns(), horizon=12
        ),
        'lstm': predict_lstm_future_recursive(
            l_model, df, (mean, std), LSTM_BEST_PARAMS, horizon=12
        ),
    }

    payload = {
        'test_dates': test['date'].tolist(),
        'test_actuals': test['ppi_index'].to_numpy(dtype=float),
        'metrics': {
            'prophet': p_metrics,
            'xgboost': x_metrics,
            'lstm': l_metrics,
        },
        'backtest': {
            'prophet': np.asarray(p_pred, dtype=float),
            'xgboost': np.asarray(x_preds, dtype=float),
            'lstm': np.asarray(l_preds, dtype=float),
        },
        'future': future,
    }
    for key in future:
        assert len(future[key]) == 12, (
            f'future forecast for {key} must have 12 points, '
            f'got {len(future[key])}'
        )
    return payload


# --------------------------------------------------------------------------
def _future_chart(df: pd.DataFrame, payload: Dict,
                  selected: list) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ppi_index'],
        mode='lines+markers',
        line={'color': X.ACTUAL_COLOR, 'width': 1.8},
        marker={'size': 3.5, 'color': X.ACTUAL_COLOR},
        name='PPI index (actual)',
        hovertemplate='%{x|%Y-%m}<br>PPI index %{y:.2f}<extra></extra>',
    ))

    for key in selected:
        pts = payload['future'][key]
        fig.add_trace(go.Scatter(
            x=[p['date'] for p in pts],
            y=[p['predicted_ppi'] for p in pts],
            mode='lines+markers',
            line={'color': X.MODEL_COLOR[key], 'width': 1.7, 'dash': 'dash'},
            marker={'size': 4.5, 'color': X.MODEL_COLOR[key]},
            name=f'{MODEL_LABELS[key]} — 2026 demo forecast',
            hovertemplate='%{x|%Y-%m}<br>%{y:.2f}<extra></extra>',
        ))

    fig.add_vline(
        x='2025-12-15', line_dash='dot', line_color='#64748b', line_width=1,
        annotation_text='2025-12 boundary — demo forecast starts',
        annotation_position='top left',
        annotation_font_size=10.5, annotation_font_color='#64748b',
    )
    X.style_figure(fig, legend=True, height=460,
                   x_title='Month', y_title='PPI index')
    X.show_chart(fig, height=460)


def _backtest_chart(payload: Dict) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=payload['test_dates'], y=payload['test_actuals'],
        mode='lines+markers',
        line={'color': X.ACTUAL_COLOR, 'width': 2},
        marker={'size': 4, 'color': X.ACTUAL_COLOR},
        name='Actual (official data)',
        hovertemplate='%{x|%Y-%m}<br>Actual %{y:.2f}<extra></extra>',
    ))
    for key in ('prophet', 'xgboost', 'lstm'):
        fig.add_trace(go.Scatter(
            x=payload['test_dates'], y=payload['backtest'][key],
            mode='lines',
            line={'color': X.MODEL_COLOR[key], 'width': 1.6},
            name=f'{MODEL_LABELS[key]} (backtest)',
            hovertemplate='%{x|%Y-%m}<br>%{y:.2f}<extra></extra>',
        ))
    X.style_figure(fig, legend=True, height=360,
                   x_title='Month (2024-01 → 2025-12)',
                   y_title='PPI index')
    X.show_chart(fig, height=360)


def _demo_metrics_frame(payload: Dict) -> pd.DataFrame:
    rows = []
    for key, label in MODEL_LABELS.items():
        m = payload['metrics'][key]
        rows.append({
            'Model': f'{label} (session re-run)',
            'MAE': f"{m['MAE']:.4f}",
            'RMSE': f"{m['RMSE']:.4f}",
            'MAPE %': f"{m['MAPE_pct']:.4f}%",
            'R²': f"{m['R_squared']:.4f}",
        })
    return pd.DataFrame(rows)
