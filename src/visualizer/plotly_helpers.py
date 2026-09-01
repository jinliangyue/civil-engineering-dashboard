"""
Plotly 图表模板库
功能：封装常用的可视化模板，保持风格统一
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 统一颜色方案
COLOR_PALETTE = {
    'primary': '#0f172a',
    'secondary': '#64748b',
    'accent': '#3b82f6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'background': '#fafafa',
    'border': '#e2e8f0',
}

# 多种材料的固定颜色（保证可重复）
MATERIAL_COLORS = {
    '钢筋': '#3b82f6',
    '水泥': '#ef4444',
    '混凝土': '#10b981',
    '砂石': '#f59e0b',
    '木材': '#8b5cf6',
    '砌块': '#ec4899',
    '装饰材料': '#06b6d4',
}


def get_material_color(material: str) -> str:
    """获取材料对应颜色，未知材料用主色"""
    return MATERIAL_COLORS.get(material, COLOR_PALETTE['accent'])


def line_chart(
    df: pd.DataFrame,
    x: str = 'date',
    y: str = 'price',
    color: Optional[str] = 'material',
    title: str = '价格走势',
    y_label: str = '价格',
    moving_avg_window: Optional[int] = None,
) -> go.Figure:
    """
    通用折线图模板
    """
    fig = go.Figure()
    if color and color in df.columns:
        for category in df[color].unique():
            subset = df[df[color] == category].sort_values(x)
            color_val = get_material_color(category) if color == 'material' else None
            fig.add_trace(go.Scatter(
                x=subset[x],
                y=subset[y],
                mode='lines+markers',
                name=str(category),
                line=dict(color=color_val, width=2),
                marker=dict(size=4),
            ))
            # 添加移动平均线
            if moving_avg_window and len(subset) >= moving_avg_window:
                ma = subset[y].rolling(window=moving_avg_window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=subset[x],
                    y=ma,
                    mode='lines',
                    name=f'{category} {moving_avg}月MA',
                    line=dict(color=color_val, width=1, dash='dash'),
                    opacity=0.6,
                ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[y],
            mode='lines+markers',
            line=dict(color=COLOR_PALETTE['accent'], width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    return fig


def multi_region_comparison(
    df: pd.DataFrame,
    material: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    多地区对比图（同一材料，不同地区）
    """
    material_df = df[df['material'] == material].copy()
    if material_df.empty:
        return go.Figure()
    fig = go.Figure()
    regions = material_df['region'].unique()
    for region in regions:
        subset = material_df[material_df['region'] == region].sort_values('date')
        fig.add_trace(go.Scatter(
            x=subset['date'],
            y=subset['price'],
            mode='lines+markers',
            name=str(region),
        ))
    fig.update_layout(
        title=title or f'{material} 多地区价格对比',
        xaxis_title='日期',
        yaxis_title='价格（元/吨）',
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    return fig


def monthly_boxplot(
    df: pd.DataFrame,
    material: Optional[str] = None,
    title: str = '月度分布',
) -> go.Figure:
    """
    月度分布箱线图（看季节性）
    """
    data = df.copy()
    if material:
        data = data[data['material'] == material]
    data['month'] = data['date'].dt.month
    fig = px.box(
        data,
        x='month',
        y='price',
        color='material' if not material else None,
        title=title,
        template='plotly_white',
        height=500,
    )
    fig.update_layout(xaxis_title='月份', yaxis_title='价格')
    return fig


def seasonal_heatmap(
    df: pd.DataFrame,
    material: Optional[str] = None,
    title: str = '季节性热图',
) -> go.Figure:
    """
    季节性热图（年 × 月）
    """
    data = df.copy()
    if material:
        data = data[data['material'] == material]
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    pivot = data.pivot_table(
        values='price',
        index='year',
        columns='month',
        aggfunc='mean',
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        colorbar=dict(title='价格'),
    ))
    fig.update_layout(
        title=title,
        xaxis_title='月份',
        yaxis_title='年份',
        template='plotly_white',
        height=500,
    )
    return fig


def forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str = '价格预测',
) -> go.Figure:
    """
    预测图（历史 + 预测 + 置信区间）
    """
    fig = go.Figure()
    # 历史数据
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['price'],
        mode='lines+markers',
        name='历史',
        line=dict(color=COLOR_PALETTE['accent'], width=2),
    ))
    # 预测
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='预测',
        line=dict(color=COLOR_PALETTE['danger'], width=2, dash='dash'),
    ))
    # 置信区间
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='95% 上界',
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='95% 置信区间',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(239, 68, 68, 0.2)',
    ))
    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='价格（元/吨）',
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    return fig


def volatility_bar(
    volatility_dict: Dict,
    title: str = '价格波动率',
) -> go.Figure:
    """
    波动率柱状图
    volatility_dict: {key: {'cv_pct': ..., 'material': ..., 'region': ...}}
    """
    items = list(volatility_dict.values())
    if not items:
        return go.Figure()
    labels = [f'{it["material"]}-{it["region"]}' for it in items]
    cv_values = [it['cv_pct'] for it in items]
    fig = go.Figure(data=[
        go.Bar(x=labels, y=cv_values, marker_color=COLOR_PALETTE['accent'])
    ])
    fig.update_layout(
        title=title,
        xaxis_title='材料-地区',
        yaxis_title='变异系数 CV（%）',
        template='plotly_white',
        height=400,
    )
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    title: str = '材料价格相关性',
) -> go.Figure:
    """
    多材料价格相关性热图
    """
    pivot = df.pivot_table(
        values='price',
        index='date',
        columns='material',
        aggfunc='mean',
    )
    corr = pivot.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='相关系数'),
    ))
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500,
    )
    return fig


def summary_metrics_card(metrics: Dict, title: str = '模型评估') -> go.Figure:
    """
    模型评估指标卡片
    """
    items = list(metrics.items())
    labels = [f'{k}<br>{v}' for k, v in items]
    fig = go.Figure()
    fig.add_annotation(
        text=f'<b>{title}</b><br><br>' + '<br>'.join(labels),
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
        align='center',
    )
    fig.update_layout(
        template='plotly_white',
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig