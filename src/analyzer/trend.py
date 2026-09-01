"""
趋势分析模块
功能：长期趋势 / 同比环比 / 价格变动幅度统计
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_long_term_trend(df: pd.DataFrame) -> Dict:
    """
    计算长期趋势（线性回归）
    返回: 斜率、截距、决定系数 R²、年化变动率
    """
    if df.empty or len(df) < 2:
        return {}
    df = df.sort_values('date').copy()
    # 用「距首日天数」做 X 轴
    df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
    x = df['days_from_start'].values
    y = df['price'].values
    # 线性回归
    slope, intercept = np.polyfit(x, y, 1)
    # R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    # 年化变动率（%）
    days_span = (df['date'].max() - df['date'].min()).days
    if days_span > 0 and intercept > 0:
        annual_change_rate = (slope * 365) / intercept * 100
    else:
        annual_change_rate = 0
    return {
        'slope_per_day': round(slope, 4),
        'intercept': round(intercept, 2),
        'r_squared': round(r_squared, 4),
        'annual_change_rate_pct': round(annual_change_rate, 2),
        'start_date': df['date'].min().strftime('%Y-%m-%d'),
        'end_date': df['date'].max().strftime('%Y-%m-%d'),
        'start_price': round(float(df['price'].iloc[0]), 2),
        'end_price': round(float(df['price'].iloc[-1]), 2),
        'total_change_pct': round((float(df['price'].iloc[-1]) / float(df['price'].iloc[0]) - 1) * 100, 2),
    }


def calculate_yoy_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算同比变动（Year-over-Year）
    同比 = (本期 - 去年同期) / 去年同期 × 100%
    返回: 新增 'yoy_change' 列的 DataFrame
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    # 计算去年同月价格
    df['last_year_price'] = df.groupby(['material', 'region', 'month'])['price'].shift(12)
    # 同比变动
    df['yoy_change_pct'] = ((df['price'] - df['last_year_price']) / df['last_year_price'] * 100).round(2)
    df = df.drop(columns=['last_year_price'])
    return df


def calculate_mom_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算环比变动（Month-over-Month）
    环比 = (本期 - 上期) / 上期 × 100%
    """
    df = df.sort_values(['material', 'region', 'date']).copy()
    df['prev_price'] = df.groupby(['material', 'region'])['price'].shift(1)
    df['mom_change_pct'] = ((df['price'] - df['prev_price']) / df['prev_price'] * 100).round(2)
    df = df.drop(columns=['prev_price'])
    return df


def calculate_moving_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    计算移动平均线（默认 3 个月）
    用于平滑短期波动，看长期趋势
    """
    df = df.sort_values(['material', 'region', 'date']).copy()
    df[f'ma_{window}'] = df.groupby(['material', 'region'])['price'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    ).round(2)
    return df


def calculate_volatility(df: pd.DataFrame) -> Dict:
    """
    计算价格波动率（标准差 / 均值 = 变异系数 CV）
    CV 越大 = 价格波动越剧烈
    """
    if df.empty:
        return {}
    result = {}
    for (material, region), group in df.groupby(['material', 'region']):
        key = f'{material}_{region}'
        cv = group['price'].std() / group['price'].mean() * 100
        result[key] = {
            'material': material,
            'region': region,
            'mean_price': round(float(group['price'].mean()), 2),
            'std_price': round(float(group['price'].std()), 2),
            'cv_pct': round(float(cv), 2),
            'max_price': round(float(group['price'].max()), 2),
            'min_price': round(float(group['price'].min()), 2),
            'price_range_pct': round((float(group['price'].max()) / float(group['price'].min()) - 1) * 100, 2),
        }
    return result


def analyze_all(df: pd.DataFrame) -> Dict:
    """
    完整趋势分析
    返回: 包含所有分析结果的字典
    """
    if df.empty:
        return {'status': 'empty'}
    result = {'status': 'success'}
    # 1. 长期趋势（每种材料 + 地区组合）
    long_term = {}
    for (material, region), group in df.groupby(['material', 'region']):
        key = f'{material}_{region}'
        long_term[key] = calculate_long_term_trend(group)
    result['long_term_trend'] = long_term
    # 2. 同比环比
    df_with_yoy = calculate_yoy_changes(df)
    df_with_yoy = calculate_mom_changes(df_with_yoy)
    df_with_yoy = calculate_moving_average(df_with_yoy, window=3)
    result['df_with_changes'] = df_with_yoy
    # 3. 波动率
    result['volatility'] = calculate_volatility(df)
    logger.info(f'趋势分析完成: {len(long_term)} 个材料-地区组合')
    return result


if __name__ == '__main__':
    # 测试用：从清洗后的数据加载
    import sys
    sys.path.insert(0, '..')
    from data_loader import load_all_raw
    from data_cleaner import clean_pipeline
    df_raw = load_all_raw()
    if df_raw.empty:
        print('没有数据')
    else:
        df_clean, _ = clean_pipeline(df_raw)
        result = analyze_all(df_clean)
        print('\n=== 长期趋势 ===')
        for key, info in result.get('long_term_trend', {}).items():
            print(f'{key}: 年化变动率 {info.get("annual_change_rate_pct")}%, R²={info.get("r_squared")}')
        print('\n=== 波动率 ===')
        for key, info in result.get('volatility', {}).items():
            print(f'{key}: CV={info.get("cv_pct")}%, 价格区间 {info.get("price_range_pct")}%')