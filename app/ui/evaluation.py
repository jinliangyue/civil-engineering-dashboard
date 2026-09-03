"""Model Evaluation — the locked P0.5 formal results (static only).

All figures in this page are hardcoded from docs/PROJECT_STATUS.md §4 (the
canonical research record). Nothing is recomputed, nothing is live. The
experiment: Train 2015-01~2021-12 (84) / Validation 2022-01~2023-12 (24,
weights locked here) / Final Test 2024-01~2025-12 (24).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:  # df unused — static page; kept for uniform signature
    X.page_header(
        'Model Evaluation',
        subtitle=C.FINAL_TEST_SUBTITLE,
        badges=[X.locked_badge(C.BADGE_FORMAL), X.locked_badge('Locked · not recomputed in this app')],
    )

    # --- KPI row ----------------------------------------------------------
    X.kpi_cards(
        [(k, v, c) for _, k, v, c in C.FINAL_TEST_KPIS],
        per_row=3,
    )

    # --- MAPE bar chart (Prophet excluded from bars, annotated below) -----
    X.section(
        'MAPE on the final test — by model',
        subtitle=(
            'Prophet (12.05%) and Seasonal Naive (1.29%) would stretch this '
            'axis — both are annotated below; the remaining models are '
            'ordered low → high.'
        ),
    )
    rows = [r for r in C.FINAL_TEST_ROWS if r[1] not in ('prophet', 'seasonal_naive')]
    # six bars: Prophet (12.05%) and Seasonal Naive (1.29%) are dropped — both
    # would stretch the axis and flatten the interesting 0.35–0.7% cluster;
    # their full rows live in the locked table below. Ordered ascending.
    rows.sort(key=lambda r: r[4])
    colors = [X.BAR_ACCENT if r[1] == 'ensemble' else X.BAR_MUTED for r in rows]

    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=[r[4] for r in rows], y=[r[0] for r in rows], orientation='h',
        marker_color=colors, text=[f'{r[4]:.2f}%' for r in rows],
        textposition='outside', textfont={'size': 11, 'color': '#334155'},
        cliponaxis=False,
        hovertemplate='%{y}<br>MAPE %{x:.2f}%<extra></extra>',
    ))
    fig.update_xaxes(range=[0, max(r[4] for r in rows) * 1.25])
    X.style_figure(fig, legend=False, height=350,
                   x_title='MAPE % (lower is better)')
    X.show_chart(fig, height=350)

    with st.container():
        st.caption('')  # spacing
        X.note_sm(
            'Seasonal Naive reaches 1.29% and Prophet 12.05% (with strongly '
            'negative R²) — both outside the readable range above. Their full '
            'rows are in the table below.'
        )

    # --- Full locked table ------------------------------------------------
    X.section('Full results — 8 models (locked)')
    X.show_dataframe(_locked_table(), height=300)

    # --- Interpretation box ------------------------------------------------
    X.section('How to read this page')
    st.markdown(f'<div class="p10-note">{X.esc(C.EVAL_INFO)}</div>',
                unsafe_allow_html=True)
    X.note(
        'The Ensemble weights were derived from the Validation period only '
        '(inverse-MAPE weighting, locked before the final test) — the test '
        'set participated in no training, no tuning and no weight '
        'calculation. Leakage controls are enforced by code-level boundary '
        'assertions (84/24/24 split, causal shift(1) features, scaler fitted '
        'on train only, rolling one-step-ahead prediction).'
    )
    X.note(
        'The walk-forward regime (2021–2023, higher volatility) tells a '
        'different story — see the Robustness page. The two pages are two '
        'complementary measurements, not two versions of the same number.'
    )


def _locked_table() -> pd.DataFrame:
    rows = []
    for label, key, mae, rmse, mape, r2 in C.FINAL_TEST_ROWS:
        rows.append({
            'Model': label,
            'MAE': f'{mae:.4f}',
            'RMSE': f'{rmse:.4f}',
            'MAPE %': f'{mape:.4f}%',
            'R²': f'{r2:.4f}',
        })
    return pd.DataFrame(rows)
