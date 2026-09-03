"""Overview — the 10-second page.

Tells an interviewer, at a glance: China industrial PPI, real monthly data
2015–2025, seven models, a formal out-of-sample final test, and where the
analysis lives. No live training here — everything is static.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:
    X.page_header(
        'China Industrial PPI — Time-Series Analysis & Forecasting',
        subtitle=(
            'Producer Price Index for China’s industrial sector, National '
            'Bureau of Statistics monthly release. 7 models, two evaluation '
            'regimes, strict out-of-sample protocols.'
        ),
    )

    # --- KPI row ---------------------------------------------------------
    X.kpi_cards([
        ('Observations', '132', 'real monthly points, 2015-01 – 2025-12'),
        ('Data coverage', '2015–2025', 'official NBS series, no gaps'),
        ('Models evaluated', '7', 'baselines + Prophet + XGBoost + LSTM'),
        ('Final test months', '24', 'OOS holdout 2024-01 – 2025-12'),
    ])

    # --- Main trend chart -------------------------------------------------
    X.section('PPI index, 2015–2025')
    fig = _trend_chart(df)
    X.show_chart(fig, height=430)
    X.note_sm(
        'Index basis: prior-year same month = 100. Shaded band: the '
        'out-of-sample final-test window (2024-01 – 2025-12). The 2024–2025 '
        'window is a relatively low-volatility regime — see the Formal '
        'Research Result below.'
    )

    # --- Formal research result card --------------------------------------
    st.markdown(X.divider(), unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="margin:0.4rem 0 0.1rem 0;">{X.locked_badge()}'
        f'&nbsp; Formal research result — final test</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="p10-page-sub" style="font-size:0.9rem;">'
        'Strict out-of-sample holdout over 2024-01 – 2025-12 (24 months). '
        'Locked to the verified research environment — shown statically, '
        'never recomputed in this app.'
        '</p>',
        unsafe_allow_html=True,
    )

    X.kpi_cards(
        [
            ('Ensemble · MAPE', '0.3551%', 'validation-weighted · R² 0.5664'),
            ('XGBoost · MAPE', '0.3558%', 'best single model · R² 0.5209'),
            ('Naive baseline · MAPE', '0.3667%', 'the bar to beat'),
            ('LSTM · MAPE', '0.4387%', 'PyTorch · R² 0.4381'),
        ],
        per_row=4,
    )
    st.markdown(
        f'<div class="p10-note" style="margin-top:0.7rem;">{X.esc(C.REGIME_CAVEAT)}</div>',
        unsafe_allow_html=True,
    )

    if st.button('Explore the analysis →', key='p10_cta_eval', type='primary'):
        st.session_state['nav_page'] = 'Model Evaluation'
        st.rerun()

    X.note_sm(
        'Robustness across historical regimes (walk-forward, 2021–2023): '
        'see the Robustness page. A live interactive demo that retrains these '
        'models per session — clearly separated from the locked results — is '
        'on the Forecast page.'
    )


def _trend_chart(df: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_vrect(
        x0='2024-01-01', x1='2025-12-01',
        fillcolor='#1e40af', opacity=0.05, line_width=0,
        layer='below',
    )
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ppi_index'],
        mode='lines+markers',
        line={'color': X.MODEL_COLOR['xgboost'], 'width': 1.8},
        marker={'size': 3.5, 'color': X.MODEL_COLOR['xgboost']},
        name='PPI index (actual)',
        hovertemplate='%{x|%Y-%m}<br>PPI index %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(
        y=100, line_dash='dot', line_color='#cbd5e1', line_width=1,
        annotation_text='100 = prior-year month parity',
        annotation_position='right', annotation_font_size=10.5,
        annotation_font_color='#94a3b8',
    )
    fig.add_annotation(
        x='2024-04-01', yref='paper', y=0.06,
        text='Final test window (OOS) 2024-01 → 2025-12',
        showarrow=False, xanchor='left',
        font={'size': 10.5, 'color': '#64748b'},
    )
    X.style_figure(
        fig, legend=False,
        x_title='Month', y_title='PPI index (prior-year same month = 100)',
    )
    return fig
