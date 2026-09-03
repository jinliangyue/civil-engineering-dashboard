"""About — project framing, pipeline, models, reproducibility, limitations,
and an honest note on removed historical data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as X
from . import constants as C


def render(df: pd.DataFrame) -> None:  # df unused — static page
    X.page_header('About this platform')

    # --- Framing ----------------------------------------------------------
    st.markdown(f'<p class="p10-page-sub">{C.ABOUT_INTRO}</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="p10-page-sub" style="font-size:0.88rem;">'
        'Author: jinliangyue (pen name 十八 / Eighteen) · 2026 autumn '
        'recruiting portfolio · <a href="{}" style="color:#1e40af;">GitHub</a>'
        ' · <a href="{}" style="color:#1e40af;">live app</a></p>'.format(
            X.esc(C.GITHUB_URL), X.esc(C.APP_URL)),
        unsafe_allow_html=True,
    )

    # --- Pipeline ----------------------------------------------------------
    X.section(
        'Pipeline',
        subtitle='The full chain — every stage is enforced by code-level assertions.',
    )
    st.markdown(X.pipe_row(C.PIPELINE_STEPS), unsafe_allow_html=True)

    # --- Models ------------------------------------------------------------
    X.section('Models evaluated (7)')
    for name, kind, impl in C.MODEL_CATALOG:
        st.markdown(
            f'<div class="p10-note">'
            f'<span class="p10-hl">{X.esc(name)}</span>'
            f'&nbsp;·&nbsp;<span style="color:#94a3b8;">{X.esc(kind)}</span>'
            f'&nbsp;—&nbsp;{X.esc(impl)}</div>',
            unsafe_allow_html=True,
        )

    # --- Ensemble design ---------------------------------------------------
    X.section(
        'Validation-weighted ensemble',
        subtitle=(
            'Weights are inverse validation MAPE, computed only from the '
            'Validation period (2022-01 – 2023-12) and locked before the '
            'final test — the test set never touches the weights.'
        ),
    )
    X.show_dataframe(
        pd.DataFrame(
            [{'Model': n, 'Weight': f'{w:.5f}'} for n, w in C.ENSEMBLE_WEIGHTS]
        ),
        height=230,
    )

    # --- Reproducibility ---------------------------------------------------
    X.section(
        'Reproducibility',
        subtitle=(
            'Locked research environment: Python 3.9.13 matrix '
            '(docs/ENVIRONMENT.md). The app’s own environment is a separate, '
            'documented deployment baseline — live demo numbers are not the '
            'locked research numbers.'
        ),
    )
    for desc, cmd in C.REPRO_COMMANDS:
        st.markdown(
            f'<div class="p10-note"><code>{X.esc(cmd)}</code>'
            f'&nbsp;<span style="color:#94a3b8;">— {desc}</span></div>',
            unsafe_allow_html=True,
        )

    # --- Limitations --------------------------------------------------------
    X.section('Limitations (read before citing any number)')
    for i, limitation in enumerate(C.LIMITATIONS, start=1):
        st.markdown(
            f'<div class="p10-note"><span class="p10-hl">{i}.</span> '
            f'{X.esc(limitation)}</div>',
            unsafe_allow_html=True,
        )

    # --- Honest note on removed data ---------------------------------------
    X.divider()
    X.section('Data history — what was removed and why')
    st.markdown(
        '<div class="p10-note">An earlier version of this project carried '
        '44 manually-estimated annual observations (4 industries) and older '
        'exploratory metrics computed on them. They were removed because '
        'they were not from a verifiable source and would have polluted the '
        'analysis; the platform now uses only the official 132-point monthly '
        'NBS series. Older numbers you may meet elsewhere (e.g. an ensemble '
        'MAPE of 0.241% or 2026 “forecasts” of 98.9 / 106.5 / 110.4 / 116.0) '
        'came from that removed data and are deprecated — they are not '
        'reported anywhere in this app.</div>',
        unsafe_allow_html=True,
    )
    X.note_sm(
        'Page labels: this dashboard uses English labels so the research '
        'framing survives a 10-second glance; project documents (README, '
        'docs/) mix Chinese and English.'
    )
