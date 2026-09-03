"""方法与说明 Methodology — pipeline, models, reproducibility, limitations,
and an honest note on removed historical data.

Everything here is static: the pipeline diagram, the model catalog, the
locked ensemble weights (README) and the reproducibility commands describe
what the research code in src/ does and what the locked numbers mean.
"""

from __future__ import annotations

import pandas as pd

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:  # df unused — static page
    X.page_head(
        'METHODOLOGY · 方法与说明',
        '方法与说明',
        subtitle=C.METHOD_INTRO,
    )

    # --- Pipeline ------------------------------------------------------------
    X.section_head(
        kicker='PIPELINE',
        title='评估流水线',
        subtitle=(
            '每一步都由代码级边界断言强制——没有一步使用最终测试窗口的信息。'
        ),
    )
    X.steps_row(C.PIPELINE_STEPS)
    X.note_sm(
        '时间切分：特征只使用滞后项（因果 shift(1)，不引入未来）；'
        '缩放器只在训练段拟合；预测按滚动单步外推生成。'
        '集成权重从验证期（2022-01 – 2023-12）逆 MAPE 计算并锁定，'
        '最终测试窗口不参与训练、调参与权重计算。'
    )

    # --- Models ---------------------------------------------------------------
    X.section_head(
        kicker='MODEL CATALOG',
        title='被评估的模型',
        subtitle='7 个模型 + 验证加权集成（Ensemble）。',
    )
    for name, kind, impl in C.MODEL_CATALOG:
        X.block(
            f'<b>{X.esc(name)}</b>'
            f'<span style="color:#8b98a9;"> · {X.esc(kind)}</span>'
            f'<span style="color:#5c6b7f;"> — {X.esc(impl)}</span>'
        )

    # --- Ensemble design -------------------------------------------------------
    X.section_head(
        kicker='ENSEMBLE',
        title='验证加权集成（权重锁定）',
        subtitle=(
            '权重 = 验证期（2022-01 – 2023-12）逆 MAPE，最终测试前锁定——'
            '测试集从未参与权重计算。'
        ),
    )
    X.show_dataframe(
        pd.DataFrame(
            [{'Model': n, 'Weight': f'{w:.5f}'} for n, w in C.ENSEMBLE_WEIGHTS]
        ),
        height=240,
    )

    # --- Reproducibility -------------------------------------------------------
    X.section_head(
        kicker='REPRODUCIBILITY',
        title='复现命令',
        subtitle=(
            '锁定研究环境：Python 3.9.13 矩阵（docs/ENVIRONMENT.md）。'
            '应用自身是独立部署基线——页面上任何现场演示数字'
            '都不是锁定研究数字。'
        ),
    )
    for desc, cmd in C.REPRO_COMMANDS:
        X.code_line(f'$ {cmd}')
        X.note_sm(f'— {desc}')

    # --- Limitations ------------------------------------------------------------
    X.section_head(kicker='LIMITATIONS', title='局限（引用任何数字前请先读）')
    for i, limitation in enumerate(C.LIMITATIONS, start=1):
        X.block(f'<span class="p10-step-no">{i:02d}</span>'
                f'<span>{X.esc(limitation)}</span>')

    # --- Honest note on removed data --------------------------------------------
    X.divider()
    X.section_head(kicker='DATA HISTORY', title='数据历史：移除了什么、为什么')
    X.block(X.esc(C.DEPRECATED_NOTE), kind='warn')
    X.note_sm(
        '作者：jinliangyue（笔名 十八）· 2026 秋招作品集 · '
        f'<a href="{C.GITHUB_URL}" style="color:#1e40af;">GitHub</a> · '
        f'<a href="{C.APP_URL}" style="color:#1e40af;">live app</a>'
    )
