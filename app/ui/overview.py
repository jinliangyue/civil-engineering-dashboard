"""总览 Overview — the 10-second page.

One compact project header, one KPI ledger line, one dominant series chart,
then the formal research result. Everything static; no live training here.
"""

from __future__ import annotations

import pandas as pd

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:
    # --- Page head --------------------------------------------------------
    X.page_head(
        'OVERVIEW · 项目总览',
        '总览',
        subtitle=(
            '中国工业生产者出厂价格指数（PPI），国家统计局官方月度发布。'
            '132 条真实观测，7 个模型、两套评估协议、严格的样本外流程。'
        ),
    )

    # --- KPI ledger strip --------------------------------------------------
    latest = df.sort_values('date').iloc[-1]
    X.kpi_strip([
        ('LATEST PPI', f'{float(latest["ppi_index"]):.2f}',
         f'指数 · {latest["date"].strftime("%Y-%m")} 官方发布'),
        ('YOY CHANGE', f'{float(latest["yoy_pct"]):.2f}%',
         '同比 · 上年同月 = 100 口径'),
        ('OBSERVATIONS', '132', '2015-01 – 2025-12 · 逐月无缺口'),
        ('FINAL TEST MAPE', '0.3551%', '集成模型 · 2024–2025 样本外', 'accent'),
    ])

    # --- Dominant trend chart ----------------------------------------------
    X.section_head(kicker='SERIES', title='PPI 指数，2015–2025')
    X.show_chart(_trend_chart(df), height=460)
    X.note_sm(
        '指数口径：上年同月 = 100。灰影区为样本外最终测试窗口'
        '（2024-01 – 2025-12），该窗口属于低波动行情区间'
        '（见下方正式研究结果）。'
    )

    # --- Formal research result --------------------------------------------
    X.section_head(
        kicker='FORMAL RESEARCH RESULT · 锁定正式结果',
        title='最终测试 · 严格样本外留出（2024-01 – 2025-12，24 个月）',
        badges=[X.locked_badge()],
    )
    X.kpi_strip([
        (C.FINAL_TEST_KPIS[0][1], C.FINAL_TEST_KPIS[0][2],
         C.FINAL_TEST_KPIS[0][3], 'accent'),
        (C.FINAL_TEST_KPIS[1][1], C.FINAL_TEST_KPIS[1][2],
         C.FINAL_TEST_KPIS[1][3]),
        ('Naive', '0.3667%', '基线对照 · 该窗口内已接近上限'),
        (C.FINAL_TEST_KPIS[2][1], C.FINAL_TEST_KPIS[2][2],
         C.FINAL_TEST_KPIS[2][3]),
    ], variant='hero')
    X.block(X.esc(C.REGIME_CAVEAT))

    X.jump_button('查看完整 8 模型评估表 →', '模型评估')
    X.note_sm(
        '跨行情稳健性（滚动外推 · 2021–2023）见「稳健性检验」页；'
        '每会话重训三个模型的交互演示在「预测」页，'
        '与上述锁定结果明确分离。'
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
        annotation_text='100 = parity with prior-year month',
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
