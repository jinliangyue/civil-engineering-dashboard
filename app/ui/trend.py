"""趋势分析 Trend — descriptive analysis of the official series.

Index-level and year-over-year views with distribution blocks. All numbers
are computed on the fly from the loaded dataframe via
src.ppi_monthly.get_monthly_summary — no hardcoded values, no research
claims; this is descriptive analysis of the official 132 observations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X


def render(df: pd.DataFrame) -> None:
    # --- Page head --------------------------------------------------------
    X.page_head(
        'TREND · 历史趋势',
        '趋势分析',
        subtitle=(
            '官方序列的描述性分析：指数水平与同比变化。'
            '全部数字由已加载的 132 条观测实时计算。'
        ),
    )

    metric = st.radio(
        '指标',
        options=['PPI 指数水平（Index）', '同比变化（YoY %）'],
        index=0,
        horizontal=True,
        key='p10_trend_metric',
    )

    if metric.startswith('PPI 指数'):
        _render_index(df)
    else:
        _render_yoy(df)


# --------------------------------------------------------------------------
def _render_index(df: pd.DataFrame) -> None:
    from src.ppi_monthly import get_monthly_summary

    s = get_monthly_summary(df)
    idx = s['ppi_index']
    X.kpi_strip([
        ('LATEST', f"{idx['latest']:.2f}", '2025-12 · 官方发布'),
        ('MEAN', f"{idx['mean']:.2f}", '2015-01 – 2025-12 全样本'),
        ('MIN', f"{idx['min']:.2f}", '通缩谷底（2015–2016 / 2020）'),
        ('MAX', f"{idx['max']:.2f}", '2021 年上行峰值'),
    ])

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ppi_index'],
        mode='lines+markers',
        line={'color': X.MODEL_COLOR['xgboost'], 'width': 1.8},
        marker={'size': 3.5, 'color': X.MODEL_COLOR['xgboost']},
        name='PPI index (actual)',
        fill='tozeroy', fillcolor='rgba(29, 78, 216, 0.05)',
        hovertemplate='%{x|%Y-%m}<br>PPI index %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(
        y=100, line_dash='dot', line_color='#cbd5e1',
        annotation_text='100 = parity with prior-year month',
        annotation_position='right',
        annotation_font_size=10.5, annotation_font_color='#94a3b8',
    )
    X.style_figure(fig, legend=False, height=460,
                   x_title='Month', y_title='PPI index (prior-year same month = 100)')
    X.show_chart(fig, height=460)
    X.note(
        '指数按构造贴近 100（上年同月 = 100）：2015–2016 与 2020 两段'
        '通缩谷底，2021 年末通胀上行见顶，随后两年温和回落，'
        '进入 2024–2025 的低波动行情。'
    )

    _distribution_block(df, col='ppi_index', label='PPI index level')


# --------------------------------------------------------------------------
def _render_yoy(df: pd.DataFrame) -> None:
    from src.ppi_monthly import get_monthly_summary

    s = get_monthly_summary(df)
    yoy = s['yoy_pct']
    X.kpi_strip([
        ('LATEST YOY', f"{yoy['latest']:.2f}%", '2025-12 · 官方发布'),
        ('MEAN YOY', f"{yoy['mean']:.2f}%", '2015-01 – 2025-12 全样本'),
        ('MIN YOY', f"{yoy['min']:.2f}%", '最深通缩月'),
        ('MAX YOY', f"{yoy['max']:.2f}%", '2021 年通胀峰值'),
    ])

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
    X.style_figure(fig, legend=False, height=440,
                   x_title='Month', y_title='YoY %')
    X.show_chart(fig, height=440)
    X.note(
        '砖红色为正同比（涨价）、青绿色为负同比（降价）。三段行情：'
        '2015–2016 通缩，2017–2021 上行，2023–2025 回落并转入温和通缩。'
    )

    _distribution_block(df, col='yoy_pct', label='YoY %')


# --------------------------------------------------------------------------
def _distribution_block(df: pd.DataFrame, col: str, label: str) -> None:
    import plotly.graph_objects as go

    values = df[col].astype(float)
    mean_v, med_v = float(values.mean()), float(values.median())
    skew_v = float(values.skew())
    kurt_v = float(values.kurtosis())
    unit = 'index' if col == 'ppi_index' else 'pct'

    X.section_head(kicker='DISTRIBUTION · 分布诊断', title='月度分布')
    X.kpi_strip([
        ('MEAN', f'{mean_v:.2f}', f'{label} · 全样本均值'),
        ('MEDIAN', f'{med_v:.2f}', f'{label} · 中位数'),
        ('SKEWNESS', f'{skew_v:+.2f}', '三阶矩 · 偏斜方向'),
        ('EXCESS KURTOSIS', f'{kurt_v:+.2f}', '四阶矩 · 尾部厚度'),
    ], variant='lean')

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
                   x_title=f'{label} ({unit})', y_title='Months')
    X.show_chart(fig, height=320)

    tail = '右侧长尾' if skew_v > 0 else '左侧长尾'
    extreme = '价格急涨' if skew_v > 0 else '价格急跌'
    X.note_sm(
        f'偏度 {skew_v:+.2f} 对应{tail}：少数 {extreme} 的月份拉长分布；'
        f'超峰度 {kurt_v:+.2f} 刻画这些尾部有多厚。'
    )
