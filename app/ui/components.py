"""Shared UI building blocks — P0.11 terminal theme.

Design contract (display layer only):
- No decorative card grid. Numbers live in hairline KPI strips (.p10-strip).
- Page / section heads use EN kicker (small caps) + CN title.
- Stage banners (.p10-panel-head) separate LIVE DEMO from FORMAL sections.
- Charts are the hero: helpers only normalize chrome, never box them.

All styling is scoped to `p10-*` classes defined in styles.py. Markdown-based
helpers return raw HTML strings; callers pass unsafe_allow_html=True. Chart /
table helpers wrap streamlit's own widgets with version-compatible width
handling (width='stretch' on streamlit >= 1.49, use_container_width before).
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Optional, Sequence

import streamlit as st

from . import constants as C
from .styles import FONT_STACK, GRID, INK

# --------------------------------------------------------------------------
# Model palette — shared by every chart that shows multiple models.
# Deliberately low-saturation; no neon / purple / gradient.
# --------------------------------------------------------------------------
MODEL_COLOR = {
    'naive': '#64748b',            # slate-500
    'seasonal_naive': '#94a3b8',   # slate-400
    'ma': '#57534e',               # stone-600 (muted warm gray)
    'ses': '#b45309',              # amber-700 (muted)
    'prophet': '#0e7490',          # cyan-700 (muted teal)
    'xgboost': '#1d4ed8',          # blue-700
    'lstm': '#15803d',             # green-700 (muted)
    'ensemble': '#1e3a8a',         # blue-900 (distinct from xgboost blue)
}
ACTUAL_COLOR = '#334155'           # slate-700 — actual observations

# Eval-page bar chart: single muted tone, ensemble accented.
BAR_MUTED = '#aab3c4'
BAR_ACCENT = '#1e40af'

# YoY bars: muted brick red for non-negative YoY, muted teal for deflation.
YOY_POS = '#bc5b4a'
YOY_NEG = '#0e7490'


def esc(text) -> str:
    return _html.escape(str(text))


def badge(kind: str, text: str = None) -> str:
    """Inline badge. kind in {'locked', 'demo'}."""
    css = 'p10-badge p10-badge-locked' if kind == 'locked' else 'p10-badge p10-badge-demo'
    if text is None:
        text = C.BADGE_LOCKED_RESEARCH if kind == 'locked' else C.BADGE_LIVE_DEMO
    return f'<span class="{css}">{esc(text)}</span>'


def locked_badge(text: str = None) -> str:
    return badge('locked', text)


def demo_badge(text: str = None) -> str:
    return badge('demo', text)


# --------------------------------------------------------------------------
# Page head: EN kicker (small caps) + CN title + CN subtitle (+ badges)
# --------------------------------------------------------------------------
def page_head(kicker: str, title: str, subtitle: str = None,
              badges: Sequence[str] = ()) -> None:
    parts = [f'<div class="p10-kick">{esc(kicker)}</div>']
    parts.append('<div class="p10-title-row">')
    parts.append(f'<span class="p10-title">{esc(title)}</span>')
    if badges:
        parts.append(f'<span class="p10-title-badges">{"".join(badges)}</span>')
    parts.append('</div>')
    if subtitle:
        parts.append(f'<div class="p10-sub">{subtitle}</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def section_head(kicker: str = None, title: str = None, subtitle: str = None,
                 badges: Sequence[str] = ()) -> None:
    parts = []
    if kicker:
        parts.append(f'<div class="p10-sec-kick">{esc(kicker)}</div>')
    if title or badges:
        parts.append('<div class="p10-sec-row">')
        if title:
            parts.append(f'<span class="p10-sec-title">{esc(title)}</span>')
        for b in badges:
            parts.append(b)
        parts.append('</div>')
    if subtitle:
        parts.append(f'<div class="p10-sec-sub">{subtitle}</div>')
    if parts:
        st.markdown('<div class="p10-sec">' + ''.join(parts) + '</div>',
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------
# KPI strip — terminal ledger line, not a card row.
# cell: (label, value, caption) or (label, value, caption, mod)
#   mod in {None, 'accent', 'teal'} — tints the value.
# variant: None | 'hero' | 'lean' — scales the value type.
# --------------------------------------------------------------------------
def kpi_strip(cells: Sequence[tuple], variant: str = None) -> None:
    inner = []
    for cell in cells:
        label, value, caption = cell[0], cell[1], cell[2]
        mod = cell[3] if len(cell) > 3 else None
        mod_css = {'accent': ' p10-s-val--accent',
                   'teal': ' p10-s-val--teal'}.get(mod, '')
        inner.append(
            f'<div class="p10-strip-cell">'
            f'<div class="p10-s-label">{esc(label)}</div>'
            f'<div class="p10-s-val{mod_css}">{esc(value)}</div>'
            f'<div class="p10-s-cap">{esc(caption)}</div>'
            f'</div>'
        )
    cls = 'p10-strip' + (f' p10-strip--{variant}' if variant else '')
    st.markdown(f'<div class="{cls}">{"".join(inner)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Stage banner — separates LIVE DEMO sections from FORMAL (locked) ones.
# --------------------------------------------------------------------------
def stage_head(kind: str, title: str, badge_html: str = None,
               subtitle: str = None) -> None:
    """kind in {'demo', 'formal'}; badge_html overrides the default badge."""
    if badge_html is None:
        badge_html = demo_badge() if kind == 'demo' else locked_badge()
    cls = 'p10-panel-head p10-panel-head--' + kind
    parts = [f'<div class="{cls}">',
             '<span class="p10-panel-flag"></span>',
             f'<span class="p10-panel-title">{esc(title)}</span>',
             f'<span class="p10-panel-badges">{badge_html}</span>',
             '</div>']
    if subtitle:
        parts.append(f'<div class="p10-note-sm">{subtitle}</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Top band (dark navy terminal header shown at the top of every page)
# --------------------------------------------------------------------------
def app_band() -> None:
    chips = [
        f'<span class="p10-chip">132 OBS · <b>2015–2025</b></span>',
        '<span class="p10-chip">SOURCE · <b>国家统计局</b></span>',
        '<span class="p10-chip">运行时离线 · <b>只读 CSV</b></span>',
        f'<a class="p10-chip" href="{C.GITHUB_URL}" target="_blank">GITHUB ↗</a>',
    ]
    st.markdown(
        f'<div class="p10-band">'
        f'<div class="p10-band-l">'
        f'<div class="p10-band-title">'
        f'<span class="p10-band-cn">{esc(C.SIDEBAR_BRAND)}</span>'
        f'<span class="p10-band-en">China industrial PPI · '
        f'time-series analytics terminal</span>'
        f'</div>'
        f'<div class="p10-band-sub">官方月度数据 · 只读模式 · '
        f'研究与演示分离的评估终端</div>'
        f'</div>'
        f'<div class="p10-band-meta">{"".join(chips)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Notes, blocks, quotes, checks, footer
# --------------------------------------------------------------------------
def note(text: str) -> None:
    st.markdown(f'<div class="p10-note">{text}</div>', unsafe_allow_html=True)


def note_sm(text: str) -> None:
    st.markdown(f'<div class="p10-note-sm">{text}</div>', unsafe_allow_html=True)


def block(text: str, kind: str = None) -> None:
    """Contextual band. kind: None | 'demo' | 'warn'."""
    cls = 'p10-block' + (f' p10-block--{kind}' if kind else '')
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def quote(text: str) -> None:
    st.markdown(f'<div class="p10-quote">{text}</div>', unsafe_allow_html=True)


def check_line(text: str) -> None:
    st.markdown(
        f'<div class="p10-note" style="margin:0.28rem 0;">'
        f'<span class="p10-check">✓</span> {esc(text)}</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="p10-divider"></div>', unsafe_allow_html=True)


def demo_disclaimer_note() -> None:
    """Exact runtime disclaimer shown wherever a live retrain feeds numbers."""
    note_sm(C.DEMO_DISCLAIMER)


def code_line(text: str) -> None:
    st.markdown(f'<div class="p10-code-line">{esc(text)}</div>',
                unsafe_allow_html=True)


def steps_row(pairs: Sequence[tuple[str, str]], start: int = 1) -> None:
    """Numbered pipeline steps — (CN, EN) pairs, e.g. Methodology page."""
    parts = ['<div class="p10-steps">']
    for i, (cn, en) in enumerate(pairs, start=start):
        parts.append(
            f'<div class="p10-step">'
            f'<span class="p10-step-no">{i:02d}</span>'
            f'<div><div class="p10-step-t">{esc(cn)}</div>'
            f'<div class="p10-step-e">{esc(en)}</div></div>'
            f'</div>'
        )
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="p10-footer">{esc(C.FOOTER_LINE)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Plotly chrome
# --------------------------------------------------------------------------
def style_figure(fig, height: int = None, legend: bool = True,
                 x_title: str = None, y_title: str = None, **layout_kw):
    """Apply the uniform visual system to a plotly figure."""
    fig.update_layout(
        template='plotly_white',
        font={'family': FONT_STACK, 'size': 12, 'color': INK},
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin={'l': 40, 'r': 16, 't': 12, 'b': 34},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'left',
            'x': 0,
            'font': {'size': 11.5},
            'itemclick': 'toggle',
            'itemdoubleclick': 'toggleothers',
        } if legend else None,
        hoverlabel={'bgcolor': '#ffffff', 'bordercolor': '#cbd5e1',
                    'font': {'color': INK, 'family': FONT_STACK}},
        **layout_kw,
    )
    fig.update_xaxes(
        showgrid=False, linecolor='#c9d1dd', tickfont={'size': 11},
        title={'text': x_title, 'font': {'size': 12}} if x_title else None,
    )
    fig.update_yaxes(
        gridcolor=GRID, linecolor='#ffffff', zerolinecolor='#c9d1dd',
        tickfont={'size': 11},
        title={'text': y_title, 'font': {'size': 12}} if y_title else None,
    )
    if height:
        fig.update_layout(height=height)
    return fig


def show_chart(fig, height: int = None) -> None:
    """Plotly chart with version-compatible width handling."""
    kwargs = {}
    if height:
        kwargs['height'] = height
    try:
        st.plotly_chart(fig, width='stretch', **kwargs)
    except TypeError:  # streamlit < 1.49
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def show_dataframe(df, height: int = None, **kwargs) -> None:
    """st.dataframe with version-compatible width handling."""
    kwargs['height'] = height or 380
    try:
        st.dataframe(df, width='stretch', **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------
# Micro helpers
# --------------------------------------------------------------------------
def jump_button(label: str, page: str) -> None:
    """In-page navigation button to another sidebar page (CN label)."""
    if st.button(label, key=f'jump_to_{page}', type='secondary'):
        st.session_state['nav_page'] = page
        st.rerun()


def pills(items: Sequence[str]) -> None:
    """Inline chips with arrows (compact provenance line)."""
    parts = []
    for i, step in enumerate(items):
        if i:
            parts.append('<span class="p10-arrow">→</span>')
        parts.append(f'<span class="p10-pill">{esc(step)}</span>')
    st.markdown(f'<div style="line-height:2.15; margin:0.2rem 0;">'
                f'{"".join(parts)}</div>', unsafe_allow_html=True)
