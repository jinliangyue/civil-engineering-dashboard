"""
Streamlit 主程序 - 年度 PPI 跨行业分析仪表盘
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

from src.data_loader import load_all_raw, get_data_summary


st.set_page_config(
    page_title='中国工业 PPI 跨行业分析',
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


def ensure_data_exists():
    """
    启动时检查 data/raw/ 是否有 CSV
    如果没有（比如 Streamlit Cloud 部署时），自动运行 generate_fallback.py 生成兜底数据
    """
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    csv_files = list(raw_dir.glob('*.csv')) if raw_dir.exists() else []
    if csv_files:
        return
    # 没有数据，自动调用 generate_fallback
    import subprocess
    import sys as _sys
    script_path = Path(__file__).parent.parent / 'scripts' / 'generate_fallback.py'
    try:
        subprocess.run([_sys.executable, str(script_path)], check=True, capture_output=True, timeout=30)
    except Exception as e:
        st.warning(f'自动生成数据失败: {e}，请手动运行 python3 scripts/generate_fallback.py')


@st.cache_data
def load_data():
    ensure_data_exists()
    df_raw = load_all_raw()
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    return df


df = load_data()

if df.empty:
    st.error('⚠️ data/raw/ 下没有找到 CSV 文件，请先生成数据')
    st.code('python3 scripts/generate_fallback.py', language='bash')
    st.stop()


# 侧边栏
st.sidebar.markdown('## 📊 中国工业 PPI 分析')
st.sidebar.markdown('---')
st.sidebar.markdown('### 数据筛选')

materials = sorted(df['material'].unique())
selected_materials = st.sidebar.multiselect(
    '选择行业',
    materials,
    default=materials,
)

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
- 数据来源：国家统计局 PPI
- 行业数：4 个
- 时间跨度：2015-2025
- 技术栈：Python + Plotly + Streamlit
''')

df_filtered = df[
    (df['material'].isin(selected_materials)) &
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1])
].copy()

if df_filtered.empty:
    st.warning('当前筛选条件下没有数据')
    st.stop()


# 主页面
st.markdown('<div class="main-header">📊 中国工业 PPI 跨行业分析平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">4 大工业行业价格指数跨年度走势 + 相关性 + 预测 · 公开数据驱动</div>', unsafe_allow_html=True)

# 摘要卡片
summary = get_data_summary(df_filtered)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('数据点', f"{summary['total_rows']:,}")
with col2:
    st.metric('行业数', len(summary['materials']))
with col3:
    st.metric('时间跨度', f"{summary['date_range']['start'][:4]} - {summary['date_range']['end'][:4]}")
with col4:
    mean_price = summary['price_range']['mean']
    st.metric('平均 PPI', f'{mean_price:.1f}')

st.markdown('---')

# Tab 切换
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📈 趋势分析',
    '🔗 行业相关性',
    '📊 同比变动',
    '🔮 未来预测',
    '📋 数据说明',
])


# ============ Tab 1: 趋势分析 ============
with tab1:
    st.markdown('### 4 行业 PPI 跨年度走势')
    fig = go.Figure()
    colors = {'黑色金属冶炼': '#3b82f6', '有色金属冶炼': '#ef4444', '黑色金属矿采选': '#10b981', '有色金属矿采选': '#f59e0b'}
    for material in selected_materials:
        sub = df_filtered[df_filtered['material'] == material].sort_values('year')
        fig.add_trace(go.Scatter(
            x=sub['year'],
            y=sub['price'],
            mode='lines+markers',
            name=material,
            line=dict(color=colors.get(material, '#64748b'), width=2.5),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>年份：%{x}<br>PPI：%{y:.1f}<extra></extra>',
        ))
    fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.5, annotation_text='基准 100', annotation_position='top right')
    fig.update_layout(
        title='中国 4 大工业行业 PPI 跨年度走势（2015-2025）',
        xaxis_title='年份',
        yaxis_title='PPI 指数（上年=100）',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('### 长期趋势统计')
    rows = []
    for material in selected_materials:
        sub = df_filtered[df_filtered['material'] == material].sort_values('year')
        years = sub['year'].values
        prices = sub['price'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, prices)
        rows.append({
            '行业': material,
            '起始值': round(float(prices[0]), 1),
            '结束值': round(float(prices[-1]), 1),
            '总变动 %': round((float(prices[-1]) / float(prices[0]) - 1) * 100, 2),
            '年均变动': round(slope, 2),
            'R²': round(r_value ** 2, 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ============ Tab 2: 相关性 ============
with tab2:
    st.markdown('### 跨行业 PPI 相关性热图')
    df_pivot = df_filtered.pivot_table(values='price', index='year', columns='material')
    corr = df_pivot.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate='%{texttexttemplate}'.replace('texttemplate', 'texttemplate') + '%{text:.2f}',
        textfont=dict(size=14),
        colorbar=dict(title='相关系数'),
    ))
    fig.update_layout(
        title='4 大工业行业 PPI 相关性',
        template='plotly_white',
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('### 相关性解读')
    if len(selected_materials) >= 2:
        for i in range(len(selected_materials)):
            for j in range(i + 1, len(selected_materials)):
                m1, m2 = selected_materials[i], selected_materials[j]
                if m1 in corr.columns and m2 in corr.columns:
                    c = corr.loc[m1, m2]
                    if c > 0.85:
                        level = '极强'
                    elif c > 0.6:
                        level = '强'
                    elif c > 0.3:
                        level = '中等'
                    else:
                        level = '弱'
                    st.markdown(f'- **{m1}** vs **{m2}**：{c:.4f}（{level}相关）')


# ============ Tab 3: 同比变动 ============
with tab3:
    st.markdown('### 各行业 PPI 同比变动（%）')
    df_yoy = df_filtered.copy()
    df_yoy = df_yoy.sort_values(['material', 'year'])
    df_yoy['yoy'] = df_yoy.groupby('material')['price'].pct_change() * 100
    fig = go.Figure()
    for material in selected_materials:
        sub = df_yoy[df_yoy['material'] == material].dropna(subset=['yoy'])
        fig.add_trace(go.Bar(
            x=sub['year'],
            y=sub['yoy'],
            name=material,
            marker_color=colors.get(material, '#64748b'),
            hovertemplate='<b>%{fullData.name}</b><br>年份：%{x}<br>同比：%{y:.2f}%<extra></extra>',
        ))
    fig.add_hline(y=0, line_color='black', line_width=0.5)
    fig.update_layout(
        title='各行业 PPI 同比变动',
        xaxis_title='年份',
        yaxis_title='同比变动 %',
        template='plotly_white',
        barmode='group',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============ Tab 4: 预测 ============
with tab4:
    st.markdown('### 2026-2028 线性回归预测')
    forecast_years = st.slider('预测未来年数', 1, 5, 3)
    predictions = []
    for material in selected_materials:
        sub = df_filtered[df_filtered['material'] == material].sort_values('year')
        years = sub['year'].values
        prices = sub['price'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, prices)
        last_year = int(years[-1])
        for i in range(1, forecast_years + 1):
            future_year = last_year + i
            pred = intercept + slope * future_year
            n = len(years)
            t_val = stats.t.ppf(0.975, n - 2)
            se = std_err * np.sqrt(1 + 1/n + (future_year - years.mean()) ** 2 / np.sum((years - years.mean()) ** 2))
            predictions.append({
                '行业': material,
                '年份': future_year,
                '预测 PPI': round(pred, 2),
                '下界 95%': round(pred - t_val * se, 2),
                '上界 95%': round(pred + t_val * se, 2),
                'R²': round(r_value ** 2, 4),
            })
    df_pred = pd.DataFrame(predictions)
    st.dataframe(df_pred, use_container_width=True)
    # 可视化
    fig = go.Figure()
    for material in selected_materials:
        sub = df_filtered[df_filtered['material'] == material].sort_values('year')
        fig.add_trace(go.Scatter(
            x=sub['year'],
            y=sub['price'],
            mode='lines+markers',
            name=f'{material} (历史)',
            line=dict(color=colors.get(material, '#64748b'), width=2),
            marker=dict(size=8),
        ))
        pred_sub = df_pred[df_pred['行业'] == material]
        fig.add_trace(go.Scatter(
            x=pred_sub['年份'],
            y=pred_sub['预测 PPI'],
            mode='lines+markers',
            name=f'{material} (预测)',
            line=dict(color=colors.get(material, '#64748b'), width=2, dash='dash'),
            marker=dict(size=8, symbol='square'),
        ))
    fig.add_vline(x=year_max + 0.5, line_dash='dot', line_color='gray', opacity=0.5)
    fig.update_layout(
        title='历史 + 预测可视化',
        xaxis_title='年份',
        yaxis_title='PPI 指数（上年=100）',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============ Tab 5: 数据说明 ============
with tab5:
    st.markdown('### 数据摘要')
    st.json(summary)
    st.markdown('### 完整数据预览')
    st.dataframe(df_filtered[['date', 'material', 'price', 'unit', 'region']].sort_values(['material', 'date']), use_container_width=True)
    csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        '下载当前筛选数据（CSV）',
        csv,
        file_name='filtered_ppi_data.csv',
        mime='text/csv',
    )
    st.markdown('### 数据来源说明')
    st.markdown('''
- **数据来源**：国家统计局 - 工业生产者出厂价格指数（PPI）
- **行业覆盖**：黑色金属冶炼、有色金属冶炼、黑色金属矿采选、有色金属矿采选
- **时间范围**：2015-2025 年度数据
- **指数基准**：上年 = 100
- **指数解读**：
  - PPI > 100：当年价格高于去年
  - PPI < 100：当年价格低于去年
  - PPI = 100：当年价格与去年持平
''')


st.markdown('---')
st.markdown('''
<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>
中国工业 PPI 跨行业分析平台 · 作者：十八 · 2026 秋招简历项目<br>
技术栈：Python + pandas + Plotly + Streamlit
</div>
''', unsafe_allow_html=True)