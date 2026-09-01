"""
数据清洗模块
功能：处理缺失值 / 异常值 / 重复行 / 单位统一
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging

from data_loader import load_all_raw, save_processed, get_data_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 单位转换到「元/吨」的系数
UNIT_CONVERSION = {
    '元/吨': 1.0,
    '元/千克': 0.001,  # 1 千克 = 0.001 吨
    '元/kg': 0.001,
    '元/立方米': None,  # 体积单位需单独处理（混凝土/砂石等）
    '元/m3': None,
    '元/平方米': None,
    '元/m2': None,
    '元/块': None,
    '元/张': None,
}


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    用 IQR 方法检测异常值
    返回: 布尔 Series（True = 异常值）
    """
    if series.empty or len(series) < 4:
        return pd.Series([False] * len(series), index=series.index)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    处理缺失值
    method: 'interpolate'（线性插值）/'ffill'（前向填充）/'bfill'（后向填充）/'drop'（删除）
    """
    df = df.copy()
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        logger.info('无缺失值')
        return df
    logger.info(f'发现缺失值: {missing_count} 个')
    # 价格列特殊处理（按时间序列插值）
    if method == 'interpolate':
        df['price'] = df.groupby(['material', 'region'])['price'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
    elif method == 'ffill':
        df['price'] = df.groupby(['material', 'region'])['price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
    elif method == 'bfill':
        df['price'] = df.groupby(['material', 'region'])['price'].transform(
            lambda x: x.fillna(method='bfill').fillna(method='ffill')
        )
    elif method == 'drop':
        df = df.dropna(subset=['price'])
    remaining = df['price'].isnull().sum()
    logger.info(f'剩余缺失值: {remaining} 个')
    return df


def handle_outliers(df: pd.DataFrame, multiplier: float = 1.5, action: str = 'flag') -> pd.DataFrame:
    """
    处理异常值
    action: 'flag'（标记但保留）/'remove'（删除）/'winsorize'（缩尾到 1%/99% 分位）
    """
    df = df.copy()
    total_outliers = 0
    for (material, region), group in df.groupby(['material', 'region']):
        if len(group) < 4:
            continue
        outlier_mask = detect_outliers_iqr(group['price'], multiplier)
        n_outliers = outlier_mask.sum()
        total_outliers += n_outliers
        if n_outliers > 0:
            logger.info(f'  {material}-{region}: 发现 {n_outliers} 个异常值')
        if action == 'flag':
            df.loc[group.index, 'is_outlier'] = outlier_mask.astype(int)
        elif action == 'remove':
            df = df.drop(group[outlier_mask].index)
        elif action == 'winsorize':
            p01 = group['price'].quantile(0.01)
            p99 = group['price'].quantile(0.99)
            df.loc[group.index, 'price'] = group['price'].clip(p01, p99)
    logger.info(f'总异常值: {total_outliers} 个, 处理方式: {action}')
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """去除重复行（按 date + material + region 三个字段判断）"""
    before = len(df)
    df = df.drop_duplicates(subset=['date', 'material', 'region'], keep='last')
    after = len(df)
    if before != after:
        logger.info(f'去除重复行: {before - after} 行')
    return df


def unify_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一单位到「元/吨」（仅适用可转换的）
    体积类（立方米/平方米/块/张）保留原单位不动
    """
    df = df.copy()
    if 'unit' not in df.columns:
        logger.warning('没有 unit 列，跳过单位统一')
        return df
    converted_count = 0
    for idx, row in df.iterrows():
        unit = str(row.get('unit', '')).strip()
        if unit in UNIT_CONVERSION and UNIT_CONVERSION[unit] is not None:
            factor = UNIT_CONVERSION[unit]
            if factor != 1.0:
                df.at[idx, 'price'] = row['price'] * factor
                df.at[idx, 'unit'] = '元/吨'
                converted_count += 1
    logger.info(f'单位转换: {converted_count} 行')
    return df


def clean_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    完整清洗 pipeline
    返回: (清洗后的 DataFrame, 清洗报告)
    """
    report = {}
    logger.info('开始数据清洗...')
    if df.empty:
        logger.warning('输入数据为空')
        return df, {'status': 'empty'}
    # 1. 去重
    df = remove_duplicates(df)
    report['after_dedup'] = len(df)
    # 2. 单位统一
    df = unify_units(df)
    report['after_unit'] = len(df)
    # 3. 处理缺失值
    df = handle_missing_values(df, method='interpolate')
    report['after_missing'] = len(df)
    report['missing_filled'] = df['price'].isnull().sum() == 0
    # 4. 处理异常值（标记不删除，保留原始信息）
    df = handle_outliers(df, multiplier=1.5, action='flag')
    report['after_outlier'] = len(df)
    if 'is_outlier' in df.columns:
        report['outlier_count'] = int(df['is_outlier'].sum())
    # 5. 排序
    df = df.sort_values(['material', 'region', 'date']).reset_index(drop=True)
    report['final_rows'] = len(df)
    report['status'] = 'success'
    logger.info(f'清洗完成: 最终 {len(df)} 行')
    return df, report


def main():
    """主函数：跑完整 pipeline"""
    print('\n=== 第 1 步：加载原始数据 ===')
    df_raw = load_all_raw()
    if df_raw.empty:
        print('没有数据，请把 CSV 文件放进 data/raw/ 目录')
        return
    summary = get_data_summary(df_raw)
    print(f'加载完成: {summary["total_rows"]} 行')
    print(f'  材料: {summary["materials"]}')
    print(f'  地区: {summary["regions"]}')
    print(f'  时间: {summary["date_range"]["start"]} ~ {summary["date_range"]["end"]}')
    print('\n=== 第 2 步：数据清洗 ===')
    df_clean, report = clean_pipeline(df_raw)
    if report.get('status') == 'success':
        print(f'清洗完成: 最终 {report["final_rows"]} 行')
        if 'outlier_count' in report:
            print(f'  异常值标记: {report["outlier_count"]} 个')
    print('\n=== 第 3 步：保存清洗后数据 ===')
    save_processed(df_clean, 'all_cleaned.csv')
    print('已保存到 data/processed/all_cleaned.csv')
    # 按材料分别保存
    for material, group in df_clean.groupby('material'):
        filename = f'{material}_cleaned.csv'
        save_processed(group, filename)
        print(f'  已保存: {filename} ({len(group)} 行)')


if __name__ == '__main__':
    main()