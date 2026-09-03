"""Data — provenance, quality checklist, raw table, CSV download."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:
    X.page_header(
        'Data',
        subtitle=(
            'One official monthly series. The app reads a committed CSV '
            'offline; nothing is fetched or estimated at runtime.'
        ),
    )

    # --- Provenance -------------------------------------------------------
    X.section('Provenance')
    st.markdown(X.pipe_row([
        'National Bureau of Statistics (NBS)',
        'akshare fetch — one-time',
        'Committed CSV',
        'Validation',
        'This app',
    ]), unsafe_allow_html=True)
    X.note_sm(
        f'File: <code>{C.DATA_FACTS["file"]}</code> · '
        f'{C.DATA_FACTS["observations"]} monthly observations '
        f'({C.DATA_FACTS["start"]} – {C.DATA_FACTS["end"]}) · UTF-8 with BOM.',
    )

    # --- Quality checklist ------------------------------------------------
    X.section('Quality checklist')
    for title, text in C.DATA_CHECKLIST:
        X.check_line(f'<span class="p10-hl">{title}.</span> {text}')

    # --- Schema -----------------------------------------------------------
    X.section('Columns')
    for name, meaning in C.DATA_COLUMNS:
        X.check_line(f'<code>{name}</code> — {meaning}')

    X.divider()

    # --- Raw table --------------------------------------------------------
    X.section(
        'Raw data',
        subtitle='First and last rows of the committed file (132 rows total).',
    )
    export = df.drop(columns=['year'], errors='ignore').copy()
    X.show_dataframe(export, height=430)

    st.download_button(
        'Download full CSV',
        data=_csv_bytes(export),
        file_name='工业PPI_全国月度_2015-2025.csv',
        mime='text/csv',
        key='p10_download_csv',
    )


def _csv_bytes(df: pd.DataFrame) -> bytes:
    # utf-8-sig matches the committed file (readable by Excel on Windows).
    return df.to_csv(index=False).encode('utf-8-sig')
