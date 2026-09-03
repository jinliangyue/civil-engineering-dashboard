"""数据 Data — provenance, coverage, quality checklist, raw table, download."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:
    # --- Page head --------------------------------------------------------
    X.page_head(
        'DATA · PROVENANCE & QUALITY',
        '数据',
        subtitle=(
            '一条官方月度序列。应用在运行时只读已提交的 CSV；'
            '不联网、不估算、不做任何 fallback。'
        ),
    )

    # --- Provenance chain --------------------------------------------------
    X.section_head(kicker='PROVENANCE', title='数据来源链路')
    X.pills([
        '国家统计局（NBS）官方发布',
        'akshare 一次性获取',
        'CSV 已提交入库',
        '列结构校验',
        '本应用离线读取',
    ])
    X.note_sm(
        f'文件：<code>{C.DATA_FACTS["file"]}</code> · '
        f'{C.DATA_FACTS["observations"]} 条月度观测 · '
        f'{C.DATA_FACTS["start"]} 至 {C.DATA_FACTS["end"]} · UTF-8 BOM 编码。'
    )

    # --- Coverage micro strip ----------------------------------------------
    X.kpi_strip([
        ('OBSERVATIONS', str(C.DATA_FACTS['observations']), '月度点 · 逐月连续'),
        ('PERIOD', f'{C.DATA_FACTS["start"]} – {C.DATA_FACTS["end"]}',
         '11 个完整自然年'),
        ('GAPS', '0', '无缺月、无插值'),
        ('SOURCE', '官方', '国家统计局 · 非估算'),
    ], variant='lean')

    # --- Quality checklist --------------------------------------------------
    X.section_head(kicker='QUALITY CHECKLIST', title='质量核对单')
    for title, text in C.DATA_CHECKLIST:
        X.check_line(f'<span class="p10-hl">{title}。</span> {text}')

    # --- Schema --------------------------------------------------------------
    X.section_head(kicker='SCHEMA', title='列结构')
    for name, meaning in C.DATA_COLUMNS:
        X.check_line(f'<code>{name}</code> — {meaning}')

    # --- Raw table ----------------------------------------------------------
    X.section_head(
        kicker='RAW TABLE',
        title='原始数据',
        subtitle='已提交文件首尾数据（共 132 行）。',
    )
    export = df.drop(columns=['year'], errors='ignore').copy()
    X.show_dataframe(export, height=430)

    st.download_button(
        '下载完整 CSV',
        data=_csv_bytes(export),
        file_name='工业PPI_全国月度_2015-2025.csv',
        mime='text/csv',
        key='p10_download_csv',
    )


def _csv_bytes(df: pd.DataFrame) -> bytes:
    # utf-8-sig matches the committed file (readable by Excel on Windows).
    return df.to_csv(index=False).encode('utf-8-sig')
