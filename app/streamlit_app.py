"""
Streamlit main — 工业 PPI 分析与预测平台 · China Industrial PPI Analytics (P0.11).

7-page information architecture (总览 / 数据 / 趋势分析 / 预测 / 模型评估 /
稳健性检验 / 方法与说明) with a hard separation between LOCKED research
results (static pages) and the LIVE demo (per-session retrain on the Forecast
page). Research code in src/ is only called, never modified; all formal
numbers are hardcoded in app/ui/constants.py from docs/PROJECT_STATUS.md
§4 / §5.

P0.11 is a pure display-layer redesign — no research logic, data, evaluation
methodology or locked numbers changed. Design: Research Analytics Terminal
(dark navy structural chrome, white workspace, restrained deep blue accent,
Chinese-first copy with English labels as terminal markers).

Author: jinliangyue (十八) · 2026 autumn recruiting portfolio.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.ppi_monthly import load_monthly_ppi

from app.ui import components as X
from app.ui import constants as C
from app.ui import styles

# 数据契约（P0.9.6 · 保持不动）：data/raw/ 下的官方月度 CSV 是唯一数据源。
# 旧年度行业 schema（date/price/material）对应的数据已在 P0.1 删除；
# 不做任何 fallback；不修改真实 CSV；不产生估算数据。路径基于 __file__。
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MONTHLY_PPI_FILE = RAW_DATA_DIR / '工业PPI_全国月度_2015-2025.csv'
REQUIRED_MONTHLY_COLUMNS = ['date', 'ppi_index', 'yoy_pct', 'ytd_index']


st.set_page_config(
    page_title=C.APP_TITLE,
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)
styles.inject()


@st.cache_data
def load_data():
    """Top-level data contract: the official monthly PPI CSV (132 points,
    national aggregate — the only data source).

    Returns (df, error):
    - error is None              → success; df has date / ppi_index / yoy_pct
                                   / ytd_index / year
    - error == 'missing'         → file absent (copy: file not found)
    - error startswith 'schema'  → file present but columns incompatible
    - error == 'unreadable'      → file present but content cannot be parsed

    File existence and required columns are checked here first, so the
    loader's exception path can never trigger an akshare network fetch.
    No fallback data is ever generated or used.
    """
    if not MONTHLY_PPI_FILE.exists():
        return pd.DataFrame(), 'missing'
    try:
        probe = pd.read_csv(MONTHLY_PPI_FILE, encoding='utf-8-sig', nrows=5)
    except Exception:
        return pd.DataFrame(), 'unreadable'
    missing = [c for c in REQUIRED_MONTHLY_COLUMNS if c not in probe.columns]
    if missing:
        return pd.DataFrame(), 'schema:' + ','.join(missing)
    df_m = load_monthly_ppi()
    if df_m.empty:
        return pd.DataFrame(), 'unreadable'
    df_m = df_m.copy()
    df_m['date'] = pd.to_datetime(df_m['date'])
    df_m = df_m.sort_values('date').reset_index(drop=True)
    df_m['year'] = df_m['date'].dt.year
    return df_m, None


df, load_error = load_data()

if load_error is not None:
    if load_error == 'missing':
        st.error('⚠️ Official monthly PPI data file not found (未找到官方月度 PPI 数据文件): '
                 'data/raw/工业PPI_全国月度_2015-2025.csv')
        st.error('Please deploy with the committed official file present in data/raw/. '
                 'This app never generates or uses fallback data '
                 '(本项目不生成、不使用任何 fallback 数据).')
    elif load_error.startswith('schema'):
        st.error('PPI data schema is incompatible with this app: the official file exists but its columns '
                 'do not match this component (missing: ' + load_error.split(':', 1)[1] + ').')
        st.error('Expected columns: date / ppi_index / yoy_pct / ytd_index. '
                 'Do not edit the real data file, and do not fabricate data to fit an old view '
                 '(请勿修改真实数据文件或伪造数据).')
    else:
        st.error('⚠️ Official monthly PPI data file exists but could not be parsed '
                 '(官方月度 PPI 数据文件存在但无法解析): data/raw/工业PPI_全国月度_2015-2025.csv')
        st.error('No fallback data exists or is generated (无任何 fallback 数据).')
    st.stop()


# --------------------------------------------------------------------------
# Sidebar — brand + page router (dark navy chrome)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="padding:0.2rem 0 0.15rem 0;">'
        f'<div class="p10-sb-cn">{X.esc(C.SIDEBAR_BRAND)}</div>'
        f'<div class="p10-sb-en">{X.esc(C.SIDEBAR_SUB)}</div>'
        f'</div>'
        f'<div class="p10-sb-rule"></div>',
        unsafe_allow_html=True,
    )
    st.radio(
        '页面',
        options=C.PAGES,
        key='nav_page',
        label_visibility='collapsed',
    )
    st.markdown(
        f'<div class="p10-sb-foot">'
        f'官方月度数据 · 132 观测 · 2015–2025<br>'
        f'<span class="p10-sb-hl">国家统计局</span> · 只读模式<br>'
        f'研究版本：P0.5 最终测试 · P0.6 滚动外推'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Page dispatch (CN label → page module). Lazy module imports keep heavy
# libs (torch / Prophet / XGBoost) out of the startup path — the Forecast
# page pulls them only inside its training function at first visit.
# --------------------------------------------------------------------------
_PAGE_MODULES = {
    '总览': 'overview',
    '数据': 'data_page',
    '趋势分析': 'trend',
    '预测': 'forecast',
    '模型评估': 'evaluation',
    '稳健性检验': 'robustness',
    '方法与说明': 'methodology',
}


def _render_page(page: str, df: pd.DataFrame) -> None:
    X.app_band()
    mod = importlib.import_module(f'app.ui.{_PAGE_MODULES[page]}')
    mod.render(df)


_render_page(st.session_state['nav_page'], df)
X.footer()
