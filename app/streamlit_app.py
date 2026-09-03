"""
Streamlit 主程序 - 中国工业 PPI 月度分析与预测（官方月度数据契约 · P0.9.6）
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from src.ppi_monthly import load_monthly_ppi, get_monthly_summary

# 数据契约（P0.9.6）：data/raw/ 下的官方月度 CSV 是唯一数据源。
# 旧年度行业 schema（date/price/material）对应的数据已在 P0.1 删除，
# 顶层不再经过 src.data_loader。路径基于 __file__，与当前工作目录无关。
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MONTHLY_PPI_FILE = RAW_DATA_DIR / '工业PPI_全国月度_2015-2025.csv'
REQUIRED_MONTHLY_COLUMNS = ['date', 'ppi_index', 'yoy_pct', 'ytd_index']


st.set_page_config(
    page_title='中国工业 PPI 分析与预测',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)


st.markdown('''
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #fafafa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e2e8f0;
}
</style>
''', unsafe_allow_html=True)


@st.cache_data
def load_data():
    """顶层数据契约：官方月度 PPI CSV（132 点全国总指数，唯一数据源）。

    返回 (df, error)：
    - error 为 None      → 成功，df 含 date / ppi_index / yoy_pct / ytd_index / year
    - error == 'missing' → 文件不存在（对应文案：Official monthly PPI data file not found）
    - error 以 'schema' 开头 → 文件存在但字段与当前组件不兼容
    - error == 'unreadable' → 文件存在但内容无法解析
    先自查文件存在性与必需列再调 loader，避免 loader 的异常路径触发 akshare 联网抓取。
    不做任何 fallback；不修改真实 CSV；不产生估算数据。
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
        st.error('⚠️ 未找到官方月度 PPI 数据文件（Official monthly PPI data file not found）：data/raw/工业PPI_全国月度_2015-2025.csv')
        st.error('请确认 data/raw/ 下存在已提交的官方数据文件后重新部署或重启。本项目不生成、不使用任何 fallback 数据。')
    elif load_error.startswith('schema'):
        st.error('PPI data schema is incompatible with this app component：官方数据文件存在，但字段与当前组件不兼容（缺失字段：' + load_error.split(':', 1)[1] + '）。')
        st.error('当前组件预期字段：date / ppi_index / yoy_pct / ytd_index。请勿修改真实数据文件，也不要生成估算数据来适配旧版视图。')
    else:
        st.error('官方月度 PPI 数据文件存在但无法解析：data/raw/工业PPI_全国月度_2015-2025.csv')
    st.stop()


# 侧边栏
st.sidebar.markdown('## 📊 中国工业 PPI 分析')
st.sidebar.markdown('---')
st.sidebar.markdown('### 数据筛选')

st.sidebar.markdown('数据维度：全国工业 PPI 总指数（官方月度 · 132 点真实数据）')

year_min = int(df['year'].min())
year_max = int(df['year'].max())
year_range = st.sidebar.slider(
    '时间范围',
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

st.sidebar.markdown('---')
st.sidebar.markdown('### 项目信息')
st.sidebar.markdown('''
- 作者：十八
- 数据来源：国家统计局 PPI（官方月度）
- 数据维度：全国总指数
- 时间跨度：2015-2025（132 点）
- 技术栈：Python + Plotly + Streamlit
''')

df_filtered = df[
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1])
].copy()

if df_filtered.empty:
    st.warning('当前筛选条件下没有数据')
    st.stop()


# 主页面
st.markdown('<div class="main-header">📊 中国工业 PPI 分析与预测平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">全国工业 PPI 月度总指数（2015-2025 · 132 点官方真实数据）· 走势 + 同比 + 机器学习模型 · 公开数据驱动</div>', unsafe_allow_html=True)

# 摘要卡片（官方月度口径）
summary = get_monthly_summary(df_filtered)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('月度数据点', f"{summary['total_months']:,}")
with col2:
    st.metric('最新 PPI', f"{summary['ppi_index']['latest']:.1f}")
with col3:
    st.metric('最新同比 %', f"{summary['yoy_pct']['latest']:.1f}%")
with col4:
    st.metric('时间跨度', f"{summary['date_range']['start'][:4]} - {summary['date_range']['end'][:4]}")

st.markdown('---')

# Tab 切换
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    '📈 趋势分析',
    '🔗 行业相关性（已下线）',
    '📊 同比变动',
    '🔮 年度预测（已下线）',
    '🤖 ML 正式结果',
    '📋 数据说明',
    '🕐 月度时间序列',
])


# ============ Tab 1: 趋势分析（官方月度数据 · 全国总指数）============
with tab1:
    st.markdown('### 全国工业 PPI 月度指数走势')
    st.markdown('原 4 行业年度对比依赖已删除的手工估算数据（P0.1），该视图已下线。当前唯一官方真实序列为全国总指数月度值；指数口径为上年同月 = 100，指数 - 100 即当月同比 %。')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['ppi_index'],
        mode='lines+markers',
        name='PPI 月度指数',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4),
        hovertemplate='月份：%{x|%Y-%m}<br>PPI：%{y:.1f}<extra></extra>',
    ))
    fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.5, annotation_text='基准 100', annotation_position='top right')
    fig.update_layout(
        title='中国工业 PPI 月度指数走势（官方月度数据）',
        xaxis_title='月份',
        yaxis_title='PPI 指数（上年同月=100）',
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### 年度平均指数（月度均值聚合）')
    df_year = df_filtered.groupby('year')['ppi_index'].mean().reset_index()
    fig2 = go.Figure(go.Bar(
        x=df_year['year'],
        y=df_year['ppi_index'],
        marker_color='#3b82f6',
        hovertemplate='年份：%{x}<br>年度平均指数：%{y:.2f}<extra></extra>',
    ))
    fig2.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.5, annotation_text='基准 100')
    fig2.update_layout(
        title='各年度平均 PPI 指数（12 个月官方指数均值）',
        xaxis_title='年份',
        yaxis_title='年度平均指数（口径：上年同月=100）',
        template='plotly_white',
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('### 长期趋势统计（年度均值序列）')
    if len(df_year) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_year['year'].values, df_year['ppi_index'].values)
        st.dataframe(pd.DataFrame([{
            '起始年均': round(float(df_year['ppi_index'].iloc[0]), 2),
            '结束年均': round(float(df_year['ppi_index'].iloc[-1]), 2),
            '总变动（指数点）': round(float(df_year['ppi_index'].iloc[-1] - df_year['ppi_index'].iloc[0]), 2),
            '年均变动（指数点/年）': round(float(slope), 3),
            'R²': round(float(r_value ** 2), 4),
        }]), use_container_width=True)
        st.markdown('注：年度平均指数为 12 个月官方指数的算术平均。该统计是描述性观察，不做外推预测（实时模型回测见 Tab 7）。')
    else:
        st.markdown('当前筛选时间范围不足 2 个年份，跳过回归统计。')


# ============ Tab 2: 行业相关性（已下线）============
with tab2:
    st.markdown('### 跨行业 PPI 相关性（已下线）')
    st.markdown('该视图原本基于 4 行业 × 11 年 = 44 个手工估算年度数据点（P0.1 commit `587f9c6` 已删除）计算行业间价格相关矩阵，数据不是可验证来源，已永久删除。')
    st.markdown('删除后当前唯一官方真实序列为全国 PPI 总指数月度值，单条序列不存在第二行业维度，跨行业相关性在真实数据契约下无法计算，故本视图下线，不做替代性伪造。')
    st.markdown('替代路径：Tab 1 全国指数走势与年度均值、Tab 3 官方月度同比口径、Tab 5 正式离线实验、Tab 7 实时月度模型回测。')


# ============ Tab 3: 同比变动（官方月度 yoy_pct 直出）============
with tab3:
    st.markdown('### 全国工业 PPI 月度同比（官方发布口径，%）')
    st.markdown('原各行业年度同比柱状图基于已删除的手工估算年度数据（P0.1 commit `587f9c6`），该视图已下线。官方月度 CSV 自带 yoy_pct 列（统计局发布口径），此处直接展示，不做二次计算、不混入任何估算。')
    df_yoy = df_filtered.dropna(subset=['yoy_pct']).sort_values('date')
    fig = go.Figure()
    marker_color = ['#ef4444' if v < 0 else '#3b82f6' for v in df_yoy['yoy_pct']]
    fig.add_trace(go.Bar(
        x=df_yoy['date'],
        y=df_yoy['yoy_pct'],
        marker_color=marker_color,
        hovertemplate='月份：%{x|%Y-%m}<br>同比：%{y:.2f}%<extra></extra>',
    ))
    fig.add_hline(y=0, line_color='black', line_width=0.5)
    fig.update_layout(
        title='全国工业 PPI 月度同比变动（官方口径）',
        xaxis_title='月份',
        yaxis_title='同比 %',
        template='plotly_white',
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    if len(df_yoy) > 0:
        neg = int((df_yoy['yoy_pct'] < 0).sum())
        row_min = df_yoy.loc[df_yoy['yoy_pct'].idxmin()]
        row_max = df_yoy.loc[df_yoy['yoy_pct'].idxmax()]
        st.markdown(f'区间内同比为负的月份：{neg} 个 / {len(df_yoy)} 个月；最低同比 {row_min["yoy_pct"]:.2f}%（{row_min["date"]:%Y-%m}）；最高同比 {row_max["yoy_pct"]:.2f}%（{row_max["date"]:%Y-%m}）。红柱为负、蓝柱为正。')


# ============ Tab 4: 年度线性外推预测（已下线）============
with tab4:
    st.markdown('### 年度线性外推预测（已下线）')
    st.markdown('该视图原本基于 4 行业 × 11 年 = 44 个手工估算年度数据点做线性回归外推（P0.1 commit `587f9c6` 已删除），数据不是可验证来源，已永久删除。')
    st.markdown('真实数据契约下唯一官方序列为全国月度总指数（132 点）；对单条真实序列做无依据的年度外推不满足数据真实性原则，故本视图下线，不生成替代性伪造预测。')
    st.markdown('正式离线实验结果见 Tab 5（锁定数字）；基于真实月度数据的实时模型回测见 Tab 7（每次会话实时重跑，属 demo 数值，不是正式锁定结果）。')


# ============ Tab 5: 机器学习预测（已废弃）============
with tab5:
    st.markdown('### ⚠️ Tab 5 已废弃')
    st.markdown('此 Tab 原本基于 4 行业 × 11 年 = 44 个手工估算年度数据点。')
    st.markdown('该项目数据已于 P0.1 删除（commit `587f9c6`），改为基于 132 个月度真实 PPI 观测的 P0.5 / P0.6 实验。')
    st.markdown('如需查看月度 PPI 时序预测与 Walk-forward Validation 结果，请使用 **Tab 7 月度时间序列**。')
    st.markdown('')
    st.markdown('正式 P0.5 Final Test 结果（2024-01 ~ 2025-12, 24 月 OOS）：')
    st.markdown('- Ensemble: MAPE = **0.3551%**, R² = **0.5664**')
    st.markdown('- XGBoost: MAPE = 0.3558%')
    st.markdown('- LSTM: MAPE = 0.4387%')
    st.markdown('')
    st.markdown('正式 P0.6 Walk-forward Mean MAPE：')
    st.markdown('- Naive 1.0192% / XGBoost 1.5958% / LSTM 1.4087%')

# ============ Tab 6: 数据说明 ============
with tab6:
    st.markdown('### 数据摘要（官方月度口径）')
    st.json(summary)
    st.markdown('### 完整数据预览')
    st.dataframe(df_filtered[['date', 'ppi_index', 'yoy_pct', 'ytd_index']], use_container_width=True)
    csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        '下载当前筛选数据（CSV）',
        csv,
        file_name='filtered_ppi_monthly.csv',
        mime='text/csv',
    )
    st.markdown('### 数据来源与字段说明')
    st.markdown('''
- 数据来源：国家统计局 PPI 月度发布（经 akshare macro_china_ppi 初始抓取后提交入库；运行期本地 CSV 优先，无 fallback、无联网刷新）
- 数据文件：data/raw/工业PPI_全国月度_2015-2025.csv（132 点真实月度观测，2015-01 至 2025-12）
- 字段说明：
  - date：月份
  - ppi_index：PPI 指数（上年同月 = 100；指数 - 100 即当月同比 %）
  - yoy_pct：同比 %（统计局发布口径）
  - ytd_index：年初至今指数（上年同期 = 100）
- 行业视图说明：原 4 行业年度对比基于手工估算数据（44 点，非可验证来源，P0.1 commit `587f9c6` 已删除），依赖它的 Tab 已下线并标注。当前平台展示与分析的唯一官方序列为全国总指数月度值。
- 数字分层：Tab 5 展示的 P0.5 / P0.6 为锁定正式实验结果；Tab 7 每次会话实时重跑训练，展示数值为 demo 数值，两者并存是刻意分层，不是矛盾。
''')


# ============ Tab 7: 月度时间序列预测 ============
with tab7:
    st.markdown('### 🕐 月度 PPI 时间序列预测（实时训练 demo）')
    st.markdown('本 Tab 用官方月度数据（2015-01 至 2025-12，132 点）按正式实验同款 108/24 切分，每次会话实时重训练 Prophet / XGBoost / LSTM 三模型并做 24 个月 OOS 回测。本页数值为当前环境的 demo 复跑值，随依赖版本漂移；正式锁定实验（P0.5 Final Test / P0.6 Walk-forward）见 Tab 5，两者并存是刻意分层，不是矛盾。')

    @st.cache_data(show_spinner=False)
    def load_monthly_data():
        from src.ppi_monthly import load_monthly_ppi, get_monthly_summary
        df_m = load_monthly_ppi()
        if df_m.empty:
            return df_m, {}
        df_m['date'] = pd.to_datetime(df_m['date'])
        return df_m, get_monthly_summary(df_m)

    df_monthly, monthly_summary = load_monthly_data()

    if df_monthly.empty:
        st.error('月度数据加载失败：本地官方 CSV（data/raw/工业PPI_全国月度_2015-2025.csv）读取异常，请确认文件完整后重启。本应用不联网抓取、不使用 fallback 数据。')
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('月度数据点', f"{monthly_summary['total_months']}")
        with col2:
            st.metric('最新 PPI', f"{monthly_summary['ppi_index']['latest']:.1f}")
        with col3:
            st.metric('最新同比 %', f"{monthly_summary['yoy_pct']['latest']:.1f}%")
        with col4:
            st.metric('区间', f"{monthly_summary['date_range']['start'][:4]} - {monthly_summary['date_range']['end'][:4]}")

        # 月度趋势图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_monthly['date'], y=df_monthly['ppi_index'],
            mode='lines', name='PPI 月度',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.1)',
        ))
        fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.5, annotation_text='基准 100')
        fig.update_layout(
            title='中国工业 PPI 月度走势（2015-2025）',
            xaxis_title='日期', yaxis_title='PPI 指数（上年同月=100）',
            template='plotly_white', height=400, hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

        # 三模型训练（对齐 src/analyzer 真实 API：108/24 切分 + P0.3 锁定 LSTM 超参）
        st.markdown('#### 🤖 三模型实时训练（demo 复跑 · 每会话重跑）')
        @st.cache_data(show_spinner=False)
        def train_monthly_models(_df):
            from src.analyzer.monthly_lstm import train_all_monthly_models
            from src.analyzer.ensemble import LSTM_BEST_PARAMS
            # Streamlit 在非主线程执行脚本；macOS 下 torch 从该线程首次进入并行区时
            # 会与已加载的 libomp（prophet/xgboost 共用）嵌套冲突导致死锁（AppTest 实测挂起）。
            # 108 点训练量极小，锁单线程无性能损失，且保证 server 模式与 AppTest 一致可跑。
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # 线程池已初始化时不可改，忽略（单线程 intra-op 已足够避免嵌套）
            # 切分与正式实验同设计：Train+Validation = 108 月（至 2023-12），Final Test = 24 月（2024-01 起）
            # train_all_monthly_models 内部 _verify_data_boundary 强校验 108/24 与日期边界，违反即抛错
            df_train_val = _df[_df['date'] <= pd.Timestamp('2023-12-31')].reset_index(drop=True)
            df_test = _df[_df['date'] >= pd.Timestamp('2024-01-01')].reset_index(drop=True)
            return train_all_monthly_models(df_train_val, df_test, LSTM_BEST_PARAMS)

        with st.spinner('正在实时训练 Prophet + XGBoost + LSTM（首次约 30-90 秒）...'):
            ml_monthly = train_monthly_models(df_monthly)

        if ml_monthly.get('status') == 'success':
            # 评估对比表
            eval_rows = []
            for m in ['prophet', 'xgboost', 'lstm']:
                if m in ml_monthly:
                    met = ml_monthly[m]['metrics']
                    label = {'prophet': 'Prophet', 'xgboost': 'XGBoost', 'lstm': 'LSTM'}[m]
                    eval_rows.append({
                        '模型': label,
                        'MAE': met['MAE'],
                        'RMSE': met['RMSE'],
                        'MAPE %': met['MAPE_pct'],
                        'R²': met['R_squared'],
                        '特点': {'prophet': '加法分解 + 年度季节性', 'xgboost': '滞后 + 滚动 + 同比特征', 'lstm': '2 层 LSTM + Dropout'}[m],
                    })
            if eval_rows:
                st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)
                st.markdown('评估口径：2024-01 至 2025-12 共 24 个月滚动一步 OOS 回测，与正式实验同切分同超参。上表为当前环境 demo 复跑值，正式锁定结果见 Tab 5。')

            # ========== 超参 / 集成说明（正式实验锁定值，不在 demo 每会话重跑）==========
            st.markdown('#### 🧠 超参与集成（正式实验锁定值）')
            st.markdown('LSTM 超参来自 P0.3 正式调参实验：Train-only 网格搜索（18 组合 × 3 折时序 CV，调参数据不触碰 Validation 与 Final Test），锁定为 hidden_size=32 / num_layers=2 / dropout=0.1 / seq_length=6 / lr=0.001。本 Tab 每会话用该锁定超参实时重训练，不再重复网格搜索。')
            st.markdown('集成模型是 P0.5 正式实验（7 模型 Validation-MAPE 反比加权，含 Naive / Seasonal Naive / MA / SES 四类基准线与子进程隔离 XGBoost），需要 84/24/24 三段划分与正式实验机器，不作为每会话 demo 重跑；Validation 锁定权重与 Final Test 指标见 Tab 5。')

            # 三模型 OOS 回测图（test_pred vs test_actuals · 2024-01 ~ 2025-12）
            st.markdown('#### 三模型 OOS 回测对比（2024-01 ~ 2025-12）')
            test_dates = df_monthly.loc[df_monthly['date'] >= pd.Timestamp('2024-01-01'), 'date'].reset_index(drop=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_dates, y=ml_monthly['prophet']['test_actuals'],
                mode='lines+markers', name='实际值',
                line=dict(color='#0f172a', width=2),
                marker=dict(size=5),
            ))
            colors_pred = {'prophet': '#10b981', 'xgboost': '#3b82f6', 'lstm': '#ef4444'}
            for m in ['prophet', 'xgboost', 'lstm']:
                if m in ml_monthly and 'test_pred' in ml_monthly[m]:
                    preds = ml_monthly[m]['test_pred']
                    if len(preds) == len(test_dates):
                        fig.add_trace(go.Scatter(
                            x=test_dates, y=preds,
                            mode='lines', name=f'{m.upper()} 预测',
                            line=dict(color=colors_pred[m], width=2, dash='dash'),
                        ))
            fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.3)
            fig.update_layout(
                title='三模型 24 个月 OOS 回测（demo 复跑值）',
                xaxis_title='日期', yaxis_title='PPI 指数',
                template='plotly_white', height=500, hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True)

            # XGBoost 特征清单（当前 src 返回结构不落库特征重要性权重，展示正式特征工程清单）
            feature_cols = ml_monthly['xgboost'].get('feature_cols') or []
            if feature_cols:
                st.markdown('#### XGBoost 特征工程清单（' + str(len(feature_cols)) + ' 项）')
                st.markdown('、'.join(feature_cols))

            # 方法论
            st.markdown('#### 方法论')
            st.markdown('''
数据来源：data/raw/工业PPI_全国月度_2015-2025.csv（132 点官方真实月度观测，2015-01 至 2025-12）。akshare macro_china_ppi 仅作为初始抓取工具；运行期读已提交的本地 CSV，无联网刷新、无 fallback、无估算数据。

数字分层：本 Tab 每次会话实时重训练，展示的是当前环境下的 demo 复跑数值（随 pandas / numpy / prophet 版本漂移）；正式锁定实验结果（P0.5 Final Test / P0.6 Walk-forward，含集成权重与指标）见 Tab 5，两组数字并存是刻意分层，不是矛盾。

训练/测试划分：与正式实验同设计——2015-01 至 2023-12 为 Train + Validation（108 点），2024-01 至 2025-12 为 24 个月 OOS 回测窗口。

LSTM 超参：P0.3 正式调参锁定值（Train-only 网格搜索，18 组合 × 3 折时序 CV，数据不触碰 Validation / Final Test）：hidden_size=32 / num_layers=2 / dropout=0.1 / seq_length=6 / lr=0.001。本 Tab 直接复用锁定超参，每会话不重复调参。

特征工程（XGBoost，清单见上方）：滞后 lag1 / lag3 / lag6 / lag12、3 / 6 / 12 月滚动均值与标准差、年月季度时间特征、同比 / 环比变动。

模型说明：
- Prophet：加法分解（趋势 + 年度季节性），24 个月滚动一步 OOS
- XGBoost：因果特征工程 + 子进程隔离滚动一步预测
- LSTM：2 层 PyTorch LSTM（hidden 32），滚动一步 OOS 预测

评估提示：2024-2025 处于 PPI 低波动区间，低 MAPE 应结合 Naive baseline 一起解读；正式对比（7 模型同口径）见 Tab 5 的 P0.5 表格。
''')
        else:
            st.error('月度模型训练失败')


st.markdown('---')
st.markdown('''
<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>
中国工业 PPI 分析与预测平台 · 作者：十八 · 2026 秋招简历项目<br>
技术栈：Python + pandas + Plotly + Streamlit + XGBoost + PyTorch + Prophet
</div>
''', unsafe_allow_html=True)