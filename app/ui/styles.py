"""Design tokens and scoped CSS for the P0.10 UI.

Keep streamlit-internal overrides minimal and defensive: unknown selectors
in older/newer streamlit versions just fail silently. Content styling lives
in classes we own (p10-*), not in streamlit's testids.
"""

import streamlit as st

# ---------------------------------------------------------------- tokens
BG = '#f5f6f8'          # app background (light gray)
CARD_BG = '#ffffff'     # cards / plot paper
BORDER = '#e3e6ec'      # hairline borders
INK = '#0f172a'         # primary text (slate-900)
SUB = '#5a6472'         # secondary text (slate-ish)
FAINT = '#94a3b8'       # captions, hairlines
PRIMARY = '#1e40af'     # brand deep blue (blue-800)
PRIMARY_BRIGHT = '#1d4ed8'  # interactive / accent blue (blue-700)
GRID = '#eef0f4'        # plot grid lines

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, '
    '"Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", '
    '"Microsoft YaHei", sans-serif'
)

CSS = f"""
<style>
/* ---- global chrome (defensive: may not match in every version) ---- */
[data-testid="stAppViewContainer"] {{
    background-color: {BG};
}}
.stApp {{
    font-family: {FONT_STACK};
}}
[data-testid="stSidebar"] {{
    background-color: #ffffff;
    border-right: 1px solid {BORDER};
}}
/* tighten default block spacing a little */
.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}}

/* ---- owned classes (p10-*) ---- */
.p10-brand {{
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    color: {PRIMARY};
    text-transform: uppercase;
}}
.p10-brand-sub {{
    font-size: 0.8rem;
    color: {SUB};
    margin-top: 2px;
    letter-spacing: 0.01em;
    text-transform: none;
}}
.p10-page-title {{
    font-size: 1.85rem;
    font-weight: 700;
    color: {INK};
    line-height: 1.25;
    margin: 0 0 0.35rem 0;
}}
.p10-page-sub {{
    font-size: 0.95rem;
    color: {SUB};
    line-height: 1.55;
    margin: 0 0 0.25rem 0;
    max-width: 880px;
}}
.p10-badge {{
    display: inline-block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 999px;
    vertical-align: middle;
    white-space: nowrap;
}}
.p10-badge-locked {{
    background: #eef2ff;
    color: {PRIMARY};
    border: 1px solid #c7d2fe;
}}
.p10-badge-demo {{
    background: #fff8e6;
    color: #8a6200;
    border: 1px solid #eeda9e;
}}
.p10-kpi {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 16px 12px 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    height: 100%;
}}
.p10-kpi-value {{
    font-size: 1.55rem;
    font-weight: 650;
    letter-spacing: -0.01em;
    color: {INK};
    line-height: 1.2;
}}
.p10-kpi-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {SUB};
    margin-top: 7px;
}}
.p10-kpi-caption {{
    font-size: 0.78rem;
    color: {FAINT};
    margin-top: 2px;
}}
.p10-note {{
    font-size: 0.86rem;
    color: {SUB};
    line-height: 1.6;
}}
.p10-note-sm {{
    font-size: 0.8rem;
    color: {FAINT};
    line-height: 1.55;
}}
.p10-quote {{
    border-left: 3px solid #d7dbe3;
    padding: 1px 0 1px 14px;
    color: {SUB};
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 10px 0;
}}
.p10-quote em {{
    color: {INK};
    font-style: italic;
}}
.p10-check {{
    color: #15803d;
    font-weight: 700;
}}
.p10-divider {{
    margin: 1.4rem 0 1.1rem 0;
    border-top: 1px solid {BORDER};
}}
.p10-footer {{
    text-align: center;
    color: {FAINT};
    font-size: 0.78rem;
    line-height: 1.6;
    margin-top: 2.2rem;
}}
.p10-side-foot {{
    font-size: 0.74rem;
    color: {FAINT};
    line-height: 1.7;
    margin-top: 14px;
    border-top: 1px solid {BORDER};
    padding-top: 12px;
}}
.p10-pill {{
    display: inline-block;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    color: {INK};
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.82rem;
    margin: 2px 4px 2px 0;
    white-space: nowrap;
}}
.p10-arrow {{
    color: {FAINT};
    font-weight: 600;
    padding: 0 2px;
}}
.p10-hl {{
    color: {PRIMARY};
    font-weight: 650;
}}
</style>
"""


def inject() -> None:
    """Inject the scoped stylesheet (call once, after st.set_page_config)."""
    st.markdown(CSS, unsafe_allow_html=True)
