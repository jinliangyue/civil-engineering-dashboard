"""模型评估 Model Evaluation — the locked P0.5 formal results (static only).

All figures on this page are hardcoded from docs/PROJECT_STATUS.md §4 (the
canonical research record). Nothing is recomputed, nothing is live. The
experiment: Train 2015-01~2021-12 (84) / Validation 2022-01~2023-12 (24,
weights locked here) / Final Test 2024-01~2025-12 (24).
"""

from __future__ import annotations

import pandas as pd

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:  # df unused — static page; kept for uniform signature
    X.page_head(
        'MODEL EVALUATION · 模型评估',
        '模型评估',
        subtitle=C.FINAL_TEST_SUBTITLE,
        badges=[X.locked_badge(C.BADGE_FORMAL)],
    )

    # --- Formal final test (visual focus) ----------------------------------
    X.stage_head(
        'formal',
        '正式评估 · 严格样本外留出（2024-01 → 2025-12 · 24 个月）',
        badge_html=X.locked_badge(C.BADGE_LOCKED_RESEARCH),
    )
    X.note(
        '实验切分：训练 2015-01 – 2021-12（84 个月）/ 验证 2022-01 – 2023-12'
        '（24 个月，集成权重只在此处锁定）/ 最终测试 2024-01 – 2025-12'
        '（24 个月）。本页全部数字为静态锁定值，不在应用中重算。'
    )
    X.kpi_strip([
        (C.FINAL_TEST_KPIS[0][1], C.FINAL_TEST_KPIS[0][2],
         C.FINAL_TEST_KPIS[0][3], 'accent'),
        (C.FINAL_TEST_KPIS[1][1], C.FINAL_TEST_KPIS[1][2],
         C.FINAL_TEST_KPIS[1][3], 'accent'),
        ('Naive', '0.3667%', '基线对照 · 该窗口内已接近上限'),
        (C.FINAL_TEST_KPIS[2][1], C.FINAL_TEST_KPIS[2][2],
         C.FINAL_TEST_KPIS[2][3]),
    ], variant='hero')

    # --- MAPE bar chart -----------------------------------------------------
    X.section_head(
        kicker='BY MODEL',
        title='最终测试 MAPE · 各模型（低 → 高）',
        subtitle=(
            'Prophet（12.05%）与 Seasonal Naive（1.29%）会拉扁坐标轴：'
            '两者在下方的完整锁定表中给出数值，此处仅画可读区间内的模型。'
        ),
    )
    rows = [r for r in C.FINAL_TEST_ROWS
            if r[1] not in ('prophet', 'seasonal_naive')]
    # six bars: Prophet and Seasonal Naive dropped (see subtitle); ordered
    # ascending so the interesting 0.35–0.7% cluster stays readable.
    rows.sort(key=lambda r: r[4])
    colors = [X.BAR_ACCENT if r[1] == 'ensemble' else X.BAR_MUTED
              for r in rows]

    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=[r[4] for r in rows], y=[r[0] for r in rows], orientation='h',
        marker_color=colors,
        text=[f'{r[4]:.2f}%' for r in rows],
        textposition='outside', textfont={'size': 11, 'color': '#334155'},
        cliponaxis=False,
        hovertemplate='%{y}<br>MAPE %{x:.2f}%<extra></extra>',
    ))
    fig.update_xaxes(range=[0, max(r[4] for r in rows) * 1.25])
    X.style_figure(fig, legend=False, height=350,
                   x_title='MAPE %（越低越好 · 最终测试 24 个月）')
    X.show_chart(fig, height=350)
    X.note_sm(
        'Seasonal Naive 为 1.29%、Prophet 为 12.05%（且 R² 深度为负），'
        '两者在图外；完整行见下方锁定表。'
    )

    # --- Full locked table ---------------------------------------------------
    X.section_head(
        kicker='LOCKED TABLE',
        title='完整结果 · 8 模型（锁定）',
        subtitle='最终测试 2024-01 – 2025-12 · MAE / RMSE / MAPE / R²。',
    )
    X.show_dataframe(_locked_table(), height=320)

    # --- How to read ---------------------------------------------------------
    X.section_head(kicker='READING', title='如何阅读本页')
    X.block(X.esc(C.EVAL_INFO))
    X.note(
        '集成权重只来自验证期（按验证 MAPE 逆加权，最终测试前锁定）——'
        '测试窗口不参与训练、调参与权重计算。防泄漏由代码级边界断言强制：'
        '84/24/24 切分、因果 shift(1) 特征、scaler 仅在训练段拟合、'
        '滚动单步外推。'
    )
    X.note(
        '2021–2023 高波动行情的滚动外推讲的是另一个故事——见'
        '「稳健性检验」页。两页是互补的两组测量，不是同一个数字的两个版本。'
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
