"""Robustness — the locked P0.6 walk-forward results (static only).

Three expanding-window folds over 2021 / 2022 / 2023 — regimes that look
nothing like the calm 2024-2025 final test. All numbers are hardcoded from
the canonical records (PROJECT_STATUS §5 means/stds; README P0.6 table for
per-fold values). No walk-forward ensemble is reported, on purpose.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:  # df unused — static page
    X.page_header(
        'Robustness',
        subtitle=(
            'Walk-forward validation over three historical regimes — higher '
            'volatility than the 2024–2025 final test. Locked research '
            'results, shown statically.'
        ),
        badges=[X.locked_badge(C.BADGE_FORMAL)],
    )

    # --- Fold strip -------------------------------------------------------
    X.section('The three folds')
    cols = st.columns(3)
    for col, (name, train_txt, test_txt) in zip(cols, C.FOLDS):
        col.markdown(
            f'<div class="p10-kpi">'
            f'<div class="p10-kpi-label">{name} · expanding window</div>'
            f'<div class="p10-note" style="margin-top:6px;">{train_txt}</div>'
            f'<div class="p10-note-sm" style="margin-top:2px;">→ {test_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    X.note_sm(C.FOLD_NOISE_NOTE)

    # --- Mean MAPE KPIs ----------------------------------------------------
    X.kpi_cards([(k, v, c) for _, k, v, c in C.WF_KPIS], per_row=3)

    # --- Chart view selector ----------------------------------------------
    view = st.radio(
        'View',
        options=['Mean ± std — all 7 models', 'Per-fold — Naive / XGBoost / LSTM'],
        index=0,
        horizontal=True,
        key='p10_wf_view',
    )
    if view.startswith('Mean'):
        _mean_std_chart()
    else:
        _per_fold_chart()

    X.note_sm(X.esc(C.WF_NO_ENSEMBLE_NOTE))
    X.note_sm(
        'Per-fold values were recorded for four models (Naive, MA, XGBoost, '
        'LSTM — README P0.6 table). Seasonal Naive, SES and Prophet '
        'fold-level values were not published, so the per-fold view charts '
        'the three lead models (MA per fold: 2.29 / 2.04 / 1.12 %). '
        'Mean ± std across all 7 models: docs/PROJECT_STATUS.md §5.'
    )

    X.divider()

    # --- Verdict quotes ----------------------------------------------------
    X.section('What the walk-forward says')
    for q in C.FOOTER_QUOTES:
        X.quote(f'“{X.esc(q)}”')
    X.note(
        'Why the walk-forward means look worse than the final test: '
        '2021 was a high-volatility swing year (annual PPI range ≈ 13.2), '
        '2022 transition, 2023 decline. The 2024–2025 final test was calm. '
        'Both measurements are correct for their own regime — they answer '
        'different questions and should not be merged into a single headline '
        'number.'
    )


def _mean_std_chart() -> None:
    import plotly.graph_objects as go

    items = sorted(C.WALK_FORWARD_MEAN_STD, key=lambda r: r[2])
    labels = [r[0] for r in items]
    means = [r[2] for r in items]
    stds = [r[3] for r in items]

    fig = go.Figure(go.Bar(
        x=means, y=labels, orientation='h',
        marker_color=[X.MODEL_COLOR.get(r[1], X.BAR_MUTED) for r in items],
        error_x={'type': 'data', 'array': stds, 'color': '#64748b',
                 'thickness': 1.2, 'width': 4},
        text=[f'{m:.2f}% ± {s:.2f}' for m, s in zip(means, stds)],
        textposition='outside', textfont={'size': 10.5, 'color': '#334155'},
        cliponaxis=False,
        hovertemplate='%{y}<br>mean MAPE %{x:.2f}% ± %{customdata:.2f}%'
                      '<extra></extra>',
        customdata=stds,
    ))
    fig.update_xaxes(range=[0, max(means) * 1.15])
    X.style_figure(fig, legend=False, height=400,
                   x_title='Mean MAPE % across 3 folds (± std)')
    X.show_chart(fig, height=400)


def _per_fold_chart() -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for key in C.PER_FOLD_MODELS:
        vals = [C.PER_FOLD_MAPE[key][f] for f in C.PER_FOLD_YEARS]
        fig.add_trace(go.Bar(
            x=C.PER_FOLD_YEARS, y=vals, name={'naive': 'Naive',
                                              'xgboost': 'XGBoost',
                                              'lstm': 'LSTM'}[key],
            marker_color=X.MODEL_COLOR[key],
            text=[f'{v:.2f}%' for v in vals],
            textposition='outside', textfont={'size': 10.5, 'color': '#334155'},
            cliponaxis=False,
            hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>',
        ))
    fig.update_yaxes(range=[0, 3.6])
    X.style_figure(fig, legend=True, height=380,
                   x_title='Fold (test year)',
                   y_title='Fold MAPE %')
    X.show_chart(fig, height=380)
    X.note_sm(
        'No model is best in all three folds: Naive wins 2022 and 2023, '
        'XGBoost and LSTM alternate on 2021. That variability is the point '
        'of this page.'
    )
