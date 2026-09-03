"""稳健性检验 Robustness — locked P0.6 walk-forward diagnostics (static).

Three expanding-window folds over 2021 / 2022 / 2023 — regimes that look
nothing like the calm 2024–2025 final test. All numbers are hardcoded from
the canonical records (PROJECT_STATUS §5 means/stds; README P0.6 table for
per-fold values). No walk-forward ensemble is reported, on purpose.

Wording contract: this page is a diagnostic across historical regimes — it
never claims absolute robustness.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:  # df unused — static page
    X.page_head(
        'ROBUSTNESS · WALK-FORWARD DIAGNOSTIC',
        '稳健性检验',
        subtitle=(
            '三个历史行情上的滚动外推诊断——波动远高于 2024–2025 最终测试。'
            '锁定研究结果，静态展示；本页是诊断，不宣称绝对稳健。'
        ),
        badges=[X.locked_badge(C.BADGE_FORMAL)],
    )

    # --- The three folds ----------------------------------------------------
    X.section_head(
        kicker='FOLDS · EXPANDING WINDOW',
        title='三个折（扩张式窗口 · 滚动外推）',
        subtitle='每折只使用该时点之前的数据训练，外推随后一个完整自然年。',
    )
    cells = []
    for name, train_txt, test_txt in C.FOLDS:
        year = test_txt.replace('测试 ', '').split(' 年')[0]
        cells.append((f'FOLD {name}', year, train_txt))
    X.kpi_strip(cells, variant='lean')
    X.note_sm(C.FOLD_NOISE_NOTE)

    # --- Mean MAPE across folds ----------------------------------------------
    X.section_head(
        kicker='MEAN MAPE · 3 FOLDS',
        title='跨折均值 ± 标准差',
        subtitle='三个 12 个月测试折上的平均 MAPE（%）；跨折均值是比单折更稳的比较量。',
    )
    cells = []
    for key, label, value, caption in C.WF_KPIS:
        cells.append((label, value, caption, 'accent' if key == 'naive' else None))
    X.kpi_strip(cells)

    # --- Chart view selector ---------------------------------------------------
    view = st.radio(
        '图表视图',
        options=[
            '跨折均值 ± 标准差（全部 7 模型）',
            '逐折（Naive / XGBoost / LSTM）',
        ],
        index=0,
        horizontal=True,
        key='p10_wf_view',
    )
    if view.startswith('跨折'):
        _mean_std_chart()
    else:
        _per_fold_chart()

    X.note_sm(X.esc(C.WF_NO_ENSEMBLE_NOTE))
    X.note_sm(
        '逐折值只对四个模型做过记录（Naive、MA、XGBoost、LSTM —— README '
        'P0.6 表）：Seasonal Naive、SES 与 Prophet 未发布单折值，'
        '因此逐折视图只画三个主力模型（MA 单折 2.29 / 2.04 / 1.12 %）。'
        '全部 7 模型的均值 ± 标准差：docs/PROJECT_STATUS.md §5。'
    )

    X.divider()

    # --- What the walk-forward says -------------------------------------------
    X.section_head(kicker='DIAGNOSIS', title='滚动外推诊断结论')
    for q in C.FOOTER_QUOTES:
        X.quote(f'「{X.esc(q)}」')
    X.note(
        '为什么跨折均值明显高于最终测试：2021 是高波动摆动年'
        '（PPI 年内振幅约 13.2），2022 转向，2023 回落；2024–2025 最终测试'
        '处于低波动平稳段。两组测量各自对其所在行情成立——它们回答不同的'
        '问题，不应合并成一个头条数字。'
    )
    X.note_sm(
        '因此本平台对「稳健」一词保持克制：三折 12 个月样本只能诊断'
        '跨行情的行为差异，不能证明模型在所有行情下都稳定。'
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
                   x_title='跨折平均 MAPE %（± 标准差 · 3 折）')
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
                   x_title='折（测试年份）',
                   y_title='单折 MAPE %')
    X.show_chart(fig, height=380)
    X.note_sm(
        '没有模型在三个折中全部占优：Naive 赢下 2022 与 2023，'
        'XGBoost 与 LSTM 在 2021 上轮流领先。这种可变性正是本页要展示的。'
    )
