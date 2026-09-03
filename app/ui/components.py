"""Shared UI building blocks (badges, KPI cards, notes, chart chrome).

All styling is scoped to `p10-*` classes defined in styles.py. Markdown-based
helpers return raw HTML strings; callers pass unsafe_allow_html=True. Chart /
table helpers wrap streamlit's own widgets with version-compatible width
handling (width='stretch' on streamlit >= 1.49, use_container_width before).
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Sequence

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


def esc(text: str) -> str:
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


def page_header(title: str, subtitle: str = None, badges: Sequence[str] = ()) -> None:
    """Standard page title block. `badges` are pre-rendered badge HTML."""
    parts = [f'<div class="p10-page-title">{esc(title)}']
    for b in badges:
        parts.append(f'&nbsp;{b}')
    parts.append('</div>')
    if subtitle:
        parts.append(f'<p class="p10-page-sub">{subtitle}</p>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def kpi_cards(items: Sequence[tuple[str, str, str]], per_row: int = 4) -> None:
    """items: (label, value, caption) — all already formatted strings."""
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        cols = st.columns(len(row))
        for col, (label, value, caption) in zip(cols, row):
            col.markdown(
                f'<div class="p10-kpi">'
                f'<div class="p10-kpi-label">{esc(label)}</div>'
                f'<div class="p10-kpi-value">{esc(value)}</div>'
                f'<div class="p10-kpi-caption">{esc(caption)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def note(text: str) -> None:
    st.markdown(f'<div class="p10-note">{text}</div>', unsafe_allow_html=True)


def note_sm(text: str) -> None:
    st.markdown(f'<div class="p10-note-sm">{text}</div>', unsafe_allow_html=True)


def quote(text: str) -> None:
    st.markdown(f'<div class="p10-quote">{text}</div>', unsafe_allow_html=True)


def check_line(text: str) -> None:
    st.markdown(
        f'<div class="p10-note"><span class="p10-check">✓</span> {esc(text)}</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="p10-divider"></div>', unsafe_allow_html=True)


def demo_disclaimer_note() -> None:
    """Exact runtime disclaimer shown wherever a live retrain feeds numbers."""
    note_sm(C.DEMO_DISCLAIMER)


def section(title: str, subtitle: str = None) -> None:
    st.markdown(f'<h3 style="margin:0.4rem 0 0.1rem 0; color:{INK};">{esc(title)}</h3>',
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="p10-page-sub" style="font-size:0.9rem;">{subtitle}</p>',
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
        showgrid=False, linecolor='#d7dbe3', tickfont={'size': 11},
        title={'text': x_title, 'font': {'size': 12}} if x_title else None,
    )
    fig.update_yaxes(
        gridcolor=GRID, linecolor='#ffffff', zerolinecolor='#d7dbe3',
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
    kwargs['height'] = height or 420
    try:
        st.dataframe(df, width='stretch', **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def footer() -> None:
    st.markdown(f'<div class="p10-footer">{esc(C.FOOTER_LINE)}</div>',
                unsafe_allow_html=True)


def pipe_row(steps: Iterable[str]) -> str:
    """Pipeline chips with arrows — returns raw HTML for unsafe_allow_html."""
    parts = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="p10-arrow">→</span>')
        parts.append(f'<span class="p10-pill">{esc(step)}</span>')
    return f'<div style="line-height:2.1;">{"".join(parts)}</div>'
