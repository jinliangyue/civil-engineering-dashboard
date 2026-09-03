"""Design tokens and scoped CSS — P0.11 「Research Analytics Terminal」 theme.

Visual system (display layer only):
- Structural chrome (sidebar, top band): dark charcoal-navy #0e1526.
- Analytical workspace: white; faint gray panels only behind diagnostics.
- One restrained deep-blue accent (#1e40af family) + muted teal (#0e7490)
  reserved for secondary analytical highlights (LIVE demo, deflation).
- Charts and formal-result ledgers are the hero; no decorative cards, no
  gradients, no shadows-play, no English-only chrome.

Streamlit-internal overrides stay minimal and defensive: unknown selectors
fail silently on other versions. Owned classes (p10-*) carry the design.
"""

from string import Template

import streamlit as st

# ---------------------------------------------------------------- tokens
NAVY = '#0e1526'            # structural chrome (sidebar, top band)
NAVY_RAISED = '#141d33'     # hover / subtle raise on navy
NAVY_LINE = 'rgba(148,163,184,0.16)'   # hairline on navy
NAVY_TXT = '#a6b4cd'        # idle text on navy
NAVY_DIM = '#6f83a6'        # captions on navy
NAVY_BRIGHT = '#f5f8ff'     # strong text on navy
NAVY_ACCENT = '#7c9bf2'     # accent glyphs on navy

BG = '#ffffff'              # analytical workspace
PANEL = '#f6f8fc'           # faint block background (diagnostics only)
BORDER = '#e4e9f1'          # hairline borders on white
INK = '#14202f'             # primary text
SUB = '#5c6b7f'             # secondary text
FAINT = '#8b98a9'           # captions / muted
PRIMARY = '#1e40af'         # restrained deep blue (brand)
PRIMARY_BRIGHT = '#1d4ed8'  # interactive blue
TEAL = '#0e7490'            # muted teal (secondary highlight: demo / deflation)
GRID = '#eef1f6'            # plot grid lines
GREEN = '#15803d'           # check marks / positive
CODE_BG = '#eef2f8'

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, '
    '"Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", '
    '"Microsoft YaHei", sans-serif'
)

CSS = Template("""
<style>
/* ============ global chrome ============ */
.stApp { font-family: $FONT_STACK; }
[data-testid="stAppViewContainer"] { background-color: $BG; }
.block-container { padding-top: 0.65rem; padding-bottom: 2.2rem; }

/* ============ structural chrome: dark navy sidebar ============ */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] { background-color: $NAVY !important; }
[data-testid="stSidebar"] { border-right: 1px solid $NAVY_LINE; }
[data-testid="stSidebarContent"] { padding: 1rem 0.7rem 1.2rem 0.7rem; }
[data-testid="stSidebar"] * { font-family: $FONT_STACK; }

.p10-sb-cn {
    font-size: 1.05rem; font-weight: 750; color: $NAVY_BRIGHT;
    letter-spacing: 0.04em;
}
.p10-sb-en {
    font-size: 0.6rem; font-weight: 700; color: $NAVY_DIM;
    letter-spacing: 0.18em; text-transform: uppercase; margin-top: 3px;
}
.p10-sb-rule { border-top: 1px solid $NAVY_LINE; margin: 0.85rem 0 0.6rem; }

/* radio nav: idle gray-blue, checked = white on a deep-blue wash */
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 1px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: $NAVY_TXT; font-size: 0.92rem; line-height: 1.35;
    padding: 5px 9px; border-radius: 6px; cursor: pointer;
    display: block; margin: 1px 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(148, 163, 184, 0.08); color: #dbe4f4;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(64, 104, 212, 0.18); color: #ffffff; font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)
    [data-testid="stRadioMark"] { border-color: $NAVY_ACCENT; }

.p10-sb-foot {
    font-size: 0.72rem; color: $NAVY_DIM; line-height: 1.9;
    margin-top: 1.1rem; border-top: 1px solid $NAVY_LINE; padding-top: 0.7rem;
}
.p10-sb-foot .p10-sb-hl { color: #c3d0e6; font-weight: 600; }

/* ============ structural chrome: top band (terminal header) ============ */
.p10-band {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.5rem 1rem;
    background: $NAVY; border: 1px solid rgba(148,163,184,0.22);
    border-radius: 10px; padding: 0.68rem 1.05rem 0.72rem 1rem;
    margin-bottom: 1.05rem;
}
.p10-band-l { border-left: 2px solid $NAVY_ACCENT; padding-left: 0.8rem;
              min-width: 0; }
.p10-band-title {
    display: flex; align-items: baseline; flex-wrap: wrap;
    column-gap: 0.75rem; row-gap: 0.1rem;
}
.p10-band-cn { font-size: 1.08rem; font-weight: 750; color: $NAVY_BRIGHT;
               letter-spacing: 0.02em; }
.p10-band-en {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: $NAVY_DIM;
}
.p10-band-sub { font-size: 0.72rem; color: $NAVY_DIM; margin-top: 3px;
                line-height: 1.5; }
.p10-band-meta { display: flex; gap: 0.45rem; flex-wrap: wrap;
                 justify-content: flex-end; }
.p10-chip {
    font-size: 0.7rem; color: #9fb0c7; border: 1px solid rgba(148,163,184,0.3);
    background: rgba(148, 163, 184, 0.07); padding: 3px 9px; border-radius: 6px;
    white-space: nowrap;
}
.p10-chip b { color: #dbe6ff; font-weight: 600; }
a.p10-chip { color: #b9c8e4; text-decoration: none; }
a.p10-chip:hover { border-color: $NAVY_ACCENT; color: #ffffff; }

/* ============ page heading (compact) ============ */
.p10-kick {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: $FAINT; margin: 1.15rem 0 0.3rem;
}
.p10-title-row { display: flex; align-items: baseline; flex-wrap: wrap;
                 column-gap: 0.7rem; row-gap: 0.3rem; margin-bottom: 0.4rem; }
.p10-title { font-size: 1.45rem; font-weight: 750; color: $INK;
             letter-spacing: -0.01em; line-height: 1.28; }
.p10-title-badges { display: inline-flex; gap: 0.4rem; align-items: center;
                    font-size: 0.8rem; }
.p10-sub { font-size: 0.9rem; color: $SUB; line-height: 1.68; margin: 0 0 0.3rem;
           max-width: 900px; }

/* ============ section headings ============ */
.p10-sec { margin-top: 1.3rem; }
.p10-sec-kick {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.22em;
    text-transform: uppercase; color: $FAINT; margin: 0 0 0.15rem;
}
.p10-sec-row { display: flex; align-items: center; flex-wrap: wrap;
               column-gap: 0.6rem; margin-bottom: 0.15rem; }
.p10-sec-title { font-size: 0.97rem; font-weight: 700; color: $INK; }
.p10-sec-sub { font-size: 0.82rem; color: $SUB; line-height: 1.6; margin: 0 0 0.45rem;
               max-width: 880px; }

/* ============ badges (pills, terminal markers) ============ */
.p10-badge {
    display: inline-block; font-size: 0.6rem; font-weight: 800;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 2.5px 8px; border-radius: 999px; white-space: nowrap;
    vertical-align: middle; border: 1px solid transparent;
}
.p10-badge-locked {
    background: #eef2ff; color: $PRIMARY; border-color: #c7d2fe;
}
.p10-badge-demo {
    background: #eaf6f6; color: #0c6a7c; border-color: #b7dde0;
}

/* ============ KPI strip (terminal ledger, no card boxes) ============ */
.p10-strip {
    display: grid; grid-auto-flow: column; grid-auto-columns: 1fr;
    border-top: 1px solid $BORDER; border-bottom: 1px solid $BORDER;
    padding: 0.62rem 0 0.58rem; margin: 0.55rem 0 0.7rem;
}
.p10-strip-cell { padding: 0 1.05rem 0 0; min-width: 0; }
.p10-strip-cell + .p10-strip-cell {
    border-left: 1px solid $BORDER; padding-left: 1.15rem;
}
.p10-s-label {
    font-size: 0.59rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: $FAINT; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.p10-s-val {
    font-size: 1.34rem; font-weight: 750; color: $INK; line-height: 1.28;
    margin: 0.14rem 0 0.12rem; letter-spacing: -0.01em;
    font-variant-numeric: tabular-nums;
}
.p10-s-val--accent { color: $PRIMARY; }
.p10-s-val--teal { color: #0a5c72; }
.p10-s-val .p10-s-unit { font-size: 0.8rem; font-weight: 600; color: $SUB;
                         margin-left: 2px; letter-spacing: 0; }
.p10-s-cap { font-size: 0.7rem; color: $FAINT; line-height: 1.45;
             overflow: hidden; text-overflow: ellipsis;
             display: -webkit-box; -webkit-line-clamp: 2;
             -webkit-box-orient: vertical; }
.p10-strip--hero .p10-s-val { font-size: 1.9rem; }
.p10-strip--lean .p10-s-val { font-size: 1.08rem; }
.p10-strip--fold { grid-auto-columns: minmax(0, 1fr); }

/* ============ panels (stage separation: LIVE DEMO vs FORMAL) ============ */
.p10-panel {
    border: 1px solid $BORDER; border-radius: 8px; background: #ffffff;
    margin: 0.7rem 0 1.05rem; overflow: hidden;
}
.p10-panel-head {
    display: flex; align-items: center; flex-wrap: wrap; gap: 0.55rem;
    padding: 0.55rem 0.95rem;
    border: 1px solid $BORDER; border-radius: 8px;
    margin: 0.6rem 0 0.5rem;
}
.p10-panel-head--demo { border-color: #c6dee0; background: #f4fafb; }
.p10-panel-head--formal { border-color: #ccd7ea; background: #f5f7fc; }
.p10-panel-flag { display: inline-block; width: 8px; height: 8px;
                  border-radius: 50%; flex: none; }
.p10-panel-head--demo .p10-panel-flag { background: $TEAL; }
.p10-panel-head--formal .p10-panel-flag { background: $PRIMARY; }
.p10-panel-title { font-size: 0.92rem; font-weight: 700; color: $INK;
                   flex: 1 1 auto; min-width: 0; line-height: 1.45; }
.p10-panel-badges { margin-left: auto; display: inline-flex; gap: 0.4rem;
                    flex: none; }

/* ============ notes, blocks, quotes ============ */
.p10-note { font-size: 0.85rem; color: $SUB; line-height: 1.68; margin: 0.3rem 0; }
.p10-note-sm { font-size: 0.78rem; color: $FAINT; line-height: 1.6;
               margin: 0.3rem 0; }
.p10-block {
    background: $PANEL; border: 1px solid #e2e8f2; border-left: 3px solid $PRIMARY;
    padding: 0.6rem 0.9rem; border-radius: 2px 6px 6px 2px;
    font-size: 0.84rem; color: #3f4c5f; line-height: 1.75; margin: 0.45rem 0 0.7rem;
}
.p10-block--demo { border-left-color: $TEAL; background: #f4fafb;
                   border-color: #e0eced; }
.p10-block--warn { border-left-color: #b45309; background: #fbf7ef;
                   border-color: #ece0cd; }
.p10-quote {
    border-left: 3px solid #d7dbe3; padding: 1px 0 1px 14px;
    color: $SUB; font-size: 0.86rem; line-height: 1.7; margin: 0.5rem 0;
}
.p10-quote em { color: $INK; font-style: italic; }
.p10-check { color: $GREEN; font-weight: 800; }
.p10-hl { color: $PRIMARY; font-weight: 650; }
.p10-hl-teal { color: $TEAL; font-weight: 650; }
.p10-divider { margin: 1.35rem 0 1rem; border-top: 1px solid $BORDER; }

/* inline code */
code {
    font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.78em; background: $CODE_BG; color: #243b66;
    border-radius: 4px; padding: 0.5px 6px;
}
.p10-code-line {
    font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.76rem; background: #f4f6fb; border: 1px solid $BORDER;
    border-left: 2px solid #b9c6dd; padding: 4px 9px; border-radius: 2px 5px 5px 2px;
    margin: 3px 0; color: #22354f; white-space: nowrap; overflow-x: auto;
    display: block;
}

/* ============ pipeline chips / steps ============ */
.p10-pill {
    display: inline-block; background: #ffffff; border: 1px solid $BORDER;
    color: $INK; border-radius: 5px; padding: 3.5px 10px; font-size: 0.8rem;
    margin: 2px 4px 2px 0; white-space: nowrap;
}
.p10-arrow { color: $FAINT; font-weight: 700; padding: 0 3px; }
.p10-steps { display: flex; flex-wrap: wrap; align-items: stretch;
             gap: 0; margin: 0.45rem 0 0.5rem; }
.p10-step {
    display: flex; gap: 0.55rem; align-items: center;
    border: 1px solid $BORDER; border-left: 3px solid $PRIMARY;
    background: #ffffff; border-radius: 2px 6px 6px 2px;
    padding: 0.42rem 0.75rem 0.4rem 0.6rem; margin: 3px 0;
}
.p10-step-no { font-size: 0.62rem; font-weight: 800; color: $PRIMARY;
               font-variant-numeric: tabular-nums; }
.p10-step-t { font-size: 0.86rem; font-weight: 700; color: $INK;
              line-height: 1.3; white-space: nowrap; }
.p10-step-e { font-size: 0.62rem; color: $FAINT; letter-spacing: 0.08em;
              text-transform: uppercase; font-weight: 600; }

/* ============ footer ============ */
.p10-footer {
    text-align: center; color: $FAINT; font-size: 0.76rem; line-height: 1.7;
    margin-top: 2.4rem; border-top: 1px solid $BORDER; padding-top: 0.9rem;
}

/* ============ page radio controls (segmented feel) ============ */
[data-testid="stRadio"] label { font-size: 0.88rem; }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 0.25rem; }

/* ============ responsive: strips stack on narrow viewports ============ */
@media (max-width: 720px) {
    .p10-strip { grid-auto-flow: row; grid-template-columns: repeat(2, 1fr); }
    .p10-strip-cell { padding: 0.4rem 0; }
    .p10-strip-cell + .p10-strip-cell { border-left: none;
        border-top: 1px solid $BORDER; padding-left: 0; }
    .p10-strip-cell:nth-child(odd) { padding-right: 0.9rem; }
    .p10-band-title { flex-direction: column; align-items: flex-start; }
    .p10-band-meta { justify-content: flex-start; }
}
</style>
""")

def inject() -> None:
    """Inject the scoped stylesheet, after st.set_page_config.

    Emitted on every script run: streamlit tears down elements between
    reruns, so a module-level "written once" guard would leave every
    subsequent page unstyled (module state survives reruns; DOM elements
    do not).

    st.markdown strips <style> tags (sanitizer), so the sheet goes through
    st.html, which keeps them. Fallback to st.markdown on versions without
    st.html.
    """
    css = CSS.substitute(
        NAVY=NAVY, NAVY_RAISED=NAVY_RAISED, NAVY_LINE=NAVY_LINE,
        NAVY_TXT=NAVY_TXT, NAVY_DIM=NAVY_DIM, NAVY_BRIGHT=NAVY_BRIGHT,
        NAVY_ACCENT=NAVY_ACCENT, BG=BG, PANEL=PANEL, BORDER=BORDER,
        INK=INK, SUB=SUB, FAINT=FAINT, PRIMARY=PRIMARY,
        PRIMARY_BRIGHT=PRIMARY_BRIGHT, TEAL=TEAL, GRID=GRID, GREEN=GREEN,
        CODE_BG=CODE_BG, FONT_STACK=FONT_STACK,
    )
    if hasattr(st, 'html'):
        st.html(css)
    else:
        st.markdown(css, unsafe_allow_html=True)
