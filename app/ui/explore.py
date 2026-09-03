"""Explore — historical trends, interactive metric switch, distributions.

All KPI values are computed on the fly from the loaded dataframe via
src.ppi_monthly.get_monthly_summary (no hardcoded numbers, no research
claims — this is descriptive analysis of the official series).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X


def render(df: pd.DataFrame) -> None:
    X.page_header(
        'Explore',
        subtitle=(
            'Descriptive analysis of the official series — index level and '
            'year-over-year change. Everything here is computed from the '
            '132 loaded observations.'
        ),
    )

    metric = st.radio(
        'Metric',
        options=['PPI index', 'Year-over-year %'],
        index=0,
        horizontal=True,
        key='p10_explore_metric',
    )

    if metric == 'PPI index':
        _render_index(df)
    else:
        _render_yoy(df)


# --------------------------------------------------------------------------
def _render_index(df: pd.DataFrame) -> None:
    from src.ppi_monthly import get_monthly_summary

    s = get_monthly_summary(df)
    idx = s['ppi_index']
    X.kpi_cards([
        ('Latest index', f"{idx['latest']:.2f}", '2025-12 · official release'),
        ('Historical mean', f"{idx['mean']:.2f}", '2015-01 – 2025-12'),
        ('Historical min', f"{idx['min']:.2f}", 'deflation trough'),
        ('Historical max', f"{idx['max']:.2f}", '2021 upswing peak'),
    ])

    X.section('PPI index over time')
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ppi_index'],
        mode='lines+markers',
        line={'color': X.MODEL_COLOR['xgboost'], 'width': 1.8},
        marker={'size': 3.5, 'color': X.MODEL_COLOR['xgboost']},
        name='PPI index',
        fill='tozeroy', fillcolor='rgba(29, 78, 216, 0.05)',
        hovertemplate='%{x|%Y-%m}<br>PPI index %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(
        y=100, line_dash='dot', line_color='#cbd5e1',
        annotation_text='100 = parity with prior-year month',
        annotation_position='right',
        annotation_font_size=10.5, annotation_font_color='#94a3b8',
    )
    X.style_figure(fig, legend=False, height=400,
                   x_title='Month', y_title='PPI index')

    X.show_chart(fig)
    X.note_sm(
        'The index stays close to 100 by construction (prior-year same month '
        '= 100): deflation troughs in 2015–2016 and 2020, an inflation '
        'upswing peaking in late 2021, then two years of mild decline into '
        'the calm 2024–2025 regime.'
    )

    _distribution_block(df, col='ppi_index', label='PPI index',
                        unit='(index level)')


# --------------------------------------------------------------------------
def _render_yoy(df: pd.DataFrame) -> None:
    from src.ppi_monthly import get_monthly_summary

    s = get_monthly_summary(df)
    yoy = s['yoy_pct']
    X.kpi_cards([
        ('Latest YoY', f"{yoy['latest']:.2f}%", '2025-12 · official release'),
        ('Mean YoY', f"{yoy['mean']:.2f}%", '2015-01 – 2025-12'),
        ('Min YoY', f"{yoy['min']:.2f}%", 'deepest deflation month'),
        ('Max YoY', f"{yoy['max']:.2f}%", '2021 inflation peak'),
    ])

    X.section('Year-over-year change, by month')
    import plotly.graph_objects as go

    yoy_vals = df['yoy_pct'].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['date'], y=yoy_vals,
        marker_color=[X.YOY_POS if v >= 0 else X.YOY_NEG for v in yoy_vals],
        name='YoY %',
        hovertemplate='%{x|%Y-%m}<br>YoY %{y:.2f}%<extra></extra>',
    ))
    fig.add_hline(
        y=0, line_width=1.2, line_color='#64748b',
        annotation_text='deflation below · inflation above',
        annotation_position='top left',
        annotation_font_size=10.5, annotation_font_color='#94a3b8',
    )
    X.style_figure(fig, legend=False, height=380,
                   x_title='Month', y_title='YoY %')

    X.show_chart(fig)
    X.note_sm(
        'Positive months (brick red) are year-over-year inflation, negative '
        'months (teal) are deflation. Three distinct regimes: 2015–2016 '
        'deflation, the 2017–2021 inflation upswing, and 2023–2025 decline '
        'into mild deflation.'
    )

    _distribution_block(df, col='yoy_pct', label='YoY %',
                        unit='(percent)')


# --------------------------------------------------------------------------
def _distribution_block(df: pd.DataFrame, col: str, label: str,
                        unit: str) -> None:
    X.section('Distribution')
    import plotly.graph_objects as go

    values = df[col].astype(float)
    mean_v, med_v = float(values.mean()), float(values.median())
    skew_v = float(values.skew())
    kurt_v = float(values.kurtosis())

    X.kpi_cards([
        ('Mean', f'{mean_v:.2f}{unit}', label),
        ('Median', f'{med_v:.2f}{unit}', label),
        ('Skewness', f'{skew_v:+.2f}', 'third moment of the series'),
        ('Excess kurtosis', f'{kurt_v:+.2f}', 'fourth moment of the series'),
    ])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=22,
        marker_color='#c7d0dd',
        marker_line={'color': '#ffffff', 'width': 0.6},
        name='frequency',
    ))
    for v, lab in [(mean_v, 'mean'), (med_v, 'median')]:
        fig.add_vline(
            x=v, line_dash='dash', line_color=X.MODEL_COLOR['xgboost'],
            annotation_text=f'{lab} {v:.2f}',
            annotation_font_size=10.5,
            annotation_font_color='#64748b',
        )
    X.style_figure(fig, legend=False, height=320,
                   x_title=f'{label} {unit}', y_title='Months')
    X.show_chart(fig)

    tail = 'right tail' if skew_v > 0 else 'left tail'
    word = 'deflationary outliers' if skew_v > 0 else 'inflationary outliers'
    X.note_sm(
        f'Skewness {skew_v:+.2f} indicates a {tail}: a few months with '
        f'atypically large {word} stretch the distribution. Excess kurtosis '
        f'{kurt_v:+.2f} describes how heavy those tails are.'
    )
