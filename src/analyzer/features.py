"""
特征工程模块
功能：为机器学习模型构造增强特征
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

特征设计思路（应对样本点少的关键）：
1. 滞后特征：去年价格、前年价格（捕捉时间依赖）
2. 跨行业特征：同产业链上下游价格（捕捉产业链联动）
3. 时间特征：年份、距基准年（捕捉长期趋势）
4. 行业 one-hot 编码（区分不同行业模式）
5. 滚动统计：3 年移动平均、移动标准差（平滑波动）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 行业 one-hot 编码
INDUSTRY_LIST = ['黑色金属冶炼', '有色金属冶炼', '黑色金属矿采选', '有色金属矿采选']

# 产业链配对（用于跨行业特征）
INDUSTRY_PAIR = {
    '黑色金属冶炼': '黑色金属矿采选',
    '有色金属冶炼': '有色金属矿采选',
    '黑色金属矿采选': '黑色金属冶炼',
    '有色金属矿采选': '有色金属冶炼',
}


def add_industry_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """行业 one-hot 编码"""
    df = df.copy()
    for industry in INDUSTRY_LIST:
        df[f'is_{industry}'] = (df['material'] == industry).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2]) -> pd.DataFrame:
    """
    滞后特征：去年价格、前年价格
    lags: 滞后期数列表
    """
    df = df.copy()
    df = df.sort_values(['material', 'date']).reset_index(drop=True)
    for lag in lags:
        df[f'price_lag_{lag}'] = df.groupby('material')['price'].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [3, 5]) -> pd.DataFrame:
    """
    滚动统计特征
    windows: 窗口大小列表（年）
    """
    df = df.copy()
    df = df.sort_values(['material', 'date']).reset_index(drop=True)
    for window in windows:
        rolling = df.groupby('material')['price'].transform(
            lambda x: x.rolling(window=window, min_periods=1)
        )
        df[f'price_ma_{window}'] = rolling.mean
        df[f'price_std_{window}'] = rolling.std
    return df


def add_cross_industry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    跨行业特征：同产业链上下游价格
    例如：黑色冶炼的同一年黑色矿采选价格
    """
    df = df.copy()
    for industry, partner in INDUSTRY_PAIR.items():
        # 构建字典：(material, year) -> price
        partner_prices = df[df['material'] == partner].set_index(
            df[df['material'] == partner]['date'].dt.year
        )['price'].to_dict()
        # 在主行业的行里添加 partner_price
        col_name = f'partner_price_{partner}'
        df[col_name] = df.apply(
            lambda row: partner_prices.get(row['date'].year, np.nan) if row['material'] == industry else np.nan,
            axis=1,
        )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    时间特征
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['years_from_base'] = df['year'] - 2015
    df['is_5year_plan_start'] = ((df['year'] - 2016) % 5 == 0).astype(int)
    return df


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    构建完整的特征矩阵
    输入：原始 PPI 数据（4 行业 × 11 年）
    输出：带特征的 DataFrame
    """
    if df_raw.empty:
        return df_raw
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = add_time_features(df)
    df = add_industry_onehot(df)
    df = add_lag_features(df, lags=[1, 2])
    df = add_rolling_features(df, windows=[3])
    df = add_cross_industry_features(df)
    # 删除前 2 年的滞后行（NaN）
    df = df.dropna(subset=['price_lag_2']).reset_index(drop=True)
    logger.info(f'特征工程完成: {len(df)} 行 × {len(df.columns)} 列')
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    获取用于 ML 模型的特征列（排除价格本身和元数据）
    """
    exclude = {'date', 'price', 'material', 'unit', 'region', 'source', 'year', 'is_outlier'}
    return [col for col in df.columns if col not in exclude]


def prepare_train_test(
    df: pd.DataFrame,
    test_years: list = [2024, 2025],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    划分训练集 / 测试集
    test_years: 用于测试的年份列表
    返回: X_train, X_test, y_train, y_test
    """
    feature_cols = get_feature_columns(df)
    train = df[~df['year'].isin(test_years)]
    test = df[df['year'].isin(test_years)]
    X_train = train[feature_cols]
    y_train = train['price']
    X_test = test[feature_cols]
    y_test = test['price']
    logger.info(f'训练集: {len(X_train)} 行, 测试集: {len(X_test)} 行')
    return X_train, X_test, y_train, y_test


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
        df_features = build_features(df_clean)
        print(f'\n特征矩阵: {df_features.shape}')
        print(f'\n特征列: {get_feature_columns(df_features)}')
        print(f'\n样本:')
        print(df_features.head())