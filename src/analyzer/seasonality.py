"""
季节性分析模块
功能：检测材料价格的季节性模式（旺季/淡季）
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

季节性在工程造价里的实际意义：
- 钢筋：通常春季（3-5 月）涨价（工地开工潮）
- 水泥：夏季（6-8 月）涨价（供应紧张 + 工地赶工）
- 砂石：冬季（11-1 月）涨价（环保限产 + 运输难）
- 木材：冬季涨价（建筑保温需求）
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_monthly_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每月平均价格（用于看季节性）
    """
    df = df.copy()
    df['month'] = df['date'].dt.month
    monthly = df.groupby(['material', 'region', 'month'])['price'].agg(['mean', 'std', 'count']).reset_index()
    monthly.columns = ['material', 'region', 'month', 'mean_price', 'std_price', 'data_count']
    monthly['mean_price'] = monthly['mean_price'].round(2)
    monthly['std_price'] = monthly['std_price'].round(2)
    return monthly


def detect_peak_trough(df: pd.DataFrame) -> Dict:
    """
    检测旺季和淡季月份
    返回每种材料的旺季/淡季月份列表
    """
    if df.empty:
        return {}
    monthly = calculate_monthly_average(df)
    result = {}
    for (material, region), group in monthly.groupby(['material', 'region']):
        key = f'{material}_{region}'
        if len(group) < 12:
            continue
        # 按月平均价格排序
        sorted_by_price = group.sort_values('mean_price', ascending=False)
        # 旺季：价格最高的 3 个月
        peak_months = sorted_by_price.head(3)['month'].tolist()
        peak_months = sorted(peak_months)
        # 淡季：价格最低的 3 个月
        trough_months = sorted_by_price.tail(3)['month'].tolist()
        trough_months = sorted(trough_months)
        # 季节性强度（最高月 / 最低月）
        max_price = group['mean_price'].max()
        min_price = group['mean_price'].min()
        seasonal_strength = (max_price - min_price) / min_price * 100 if min_price > 0 else 0
        result[key] = {
            'material': material,
            'region': region,
            'peak_months': peak_months,
            'trough_months': trough_months,
            'seasonal_strength_pct': round(seasonal_strength, 2),
            'max_month_price': round(float(max_price), 2),
            'min_month_price': round(float(min_price), 2),
        }
    return result


def seasonal_decomposition(df: pd.DataFrame, period: int = 12) -> Dict:
    """
    时间序列分解（trend + seasonal + residual）
    使用 statsmodels 的 seasonal_decompose
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = {}
    for (material, region), group in df.groupby(['material', 'region']):
        if len(group) < 2 * period:
            logger.warning(f'{material}-{region} 数据不足 {2*period} 个月，跳过分解')
            continue
        key = f'{material}_{region}'
        ts = group.set_index('date')['price'].sort_index()
        ts = ts.asfreq('MS')  # 月开始
        ts = ts.interpolate()  # 缺失值插值
        try:
            decomposition = seasonal_decompose(ts, model='additive', period=period)
            result[key] = {
                'trend': decomposition.trend.dropna(),
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid.dropna(),
            }
        except Exception as e:
            logger.warning(f'{material}-{region} 分解失败: {e}')
    return result


def seasonal_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算季节性指数
    季节性指数 = 各月平均价格 / 总平均价格
    >1 表示旺季，<1 表示淡季
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['month'] = df['date'].dt.month
    # 全期平均
    overall_mean = df.groupby(['material', 'region'])['price'].mean()
    # 各月平均
    monthly_mean = df.groupby(['material', 'region', 'month'])['price'].mean()
    # 季节性指数
    seasonal_idx = (monthly_mean / overall_mean).reset_index()
    seasonal_idx.columns = ['material', 'region', 'month', 'seasonal_index']
    seasonal_idx['seasonal_index'] = seasonal_idx['seasonal_index'].round(4)
    return seasonal_idx


def analyze_all_seasonality(df: pd.DataFrame) -> Dict:
    """
    完整季节性分析
    """
    if df.empty:
        return {'status': 'empty'}
    result = {'status': 'success'}
    result['monthly_avg'] = calculate_monthly_average(df)
    result['peak_trough'] = detect_peak_trough(df)
    result['seasonal_index'] = seasonal_index(df)
    # 时间序列分解（耗时较长，按需调用）
    # result['decomposition'] = seasonal_decomposition(df)
    logger.info(f'季节性分析完成')
    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from data_loader import load_all_raw
    from data_cleaner import clean_pipeline
    df_raw = load_all_raw()
    if df_raw.empty:
        print('没有数据')
    else:
        df_clean, _ = clean_pipeline(df_raw)
        result = analyze_all_seasonality(df_clean)
        print('\n=== 旺季/淡季 ===')
        for key, info in result.get('peak_trough', {}).items():
            print(f'{key}: 旺季月份 {info["peak_months"]}, 淡季月份 {info["trough_months"]}, 季节性强度 {info["seasonal_strength_pct"]}%')
        print('\n=== 季节性指数（前 12 行） ===')
        print(result['seasonal_index'].head(12))