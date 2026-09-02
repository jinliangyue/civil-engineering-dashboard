"""
月度 PPI 数据加载模块
功能：从 data/raw/ 加载 akshare 抓取的月度 PPI 总指数
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

数据来源：akshare.macro_china_ppi()（间接从国家统计局月度发布抓取）
时间范围：2015-01 至 2025-12（132 个月度点）
字段：date / ppi_index / yoy_pct / ytd_index
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# 标准文件名
MONTHLY_FILE = RAW_DATA_DIR / '工业PPI_全国月度_2015-2025.csv'


def fetch_monthly_ppi_from_akshare(
    start_year: int = 2015,
    end_year: int = 2025,
    save: bool = True,
) -> pd.DataFrame:
    """
    通过 akshare 从国家统计局抓取月度 PPI 数据
    若 save=True 则保存到 data/raw/工业PPI_全国月度_2015-2025.csv
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error('akshare 未安装，请 pip install akshare')
        return pd.DataFrame()
    try:
        df = ak.macro_china_ppi()
        if df.empty:
            logger.error('akshare 返回空数据')
            return pd.DataFrame()
        # 数据清洗
        df['日期'] = df['月份'].str.replace('月份', '').str.replace('年', '-').str.strip()
        df = df[['日期', '当月', '当月同比增长', '累计']].copy()
        df.columns = ['date', 'ppi_index', 'yoy_pct', 'ytd_index']
        df['date'] = pd.to_datetime(df['date'] + '-01')
        # 时间筛选
        df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)].copy()
        df = df.sort_values('date').reset_index(drop=True)
        if save:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(MONTHLY_FILE, index=False, encoding='utf-8-sig')
            logger.info(f'已保存到: {MONTHLY_FILE}（{len(df)} 行）')
        return df
    except Exception as e:
        logger.error(f'akshare 抓取失败: {e}')
        return pd.DataFrame()


def load_monthly_ppi(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    加载月度 PPI 数据
    优先级：本地 CSV > akshare 在线抓取
    """
    filepath = filepath or MONTHLY_FILE
    if filepath.exists():
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f'从本地加载: {filepath.name}（{len(df)} 行）')
            return df
        except Exception as e:
            logger.warning(f'本地加载失败: {e}，尝试在线抓取')
    # 在线抓取
    return fetch_monthly_ppi_from_akshare()


def get_monthly_summary(df: pd.DataFrame) -> dict:
    """月度数据摘要"""
    if df.empty:
        return {'status': 'empty'}
    return {
        'status': 'ok',
        'total_months': len(df),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m'),
            'end': df['date'].max().strftime('%Y-%m'),
        },
        'ppi_index': {
            'min': float(df['ppi_index'].min()),
            'max': float(df['ppi_index'].max()),
            'mean': round(float(df['ppi_index'].mean()), 2),
            'latest': float(df['ppi_index'].iloc[-1]),
        },
        'yoy_pct': {
            'min': float(df['yoy_pct'].min()),
            'max': float(df['yoy_pct'].max()),
            'mean': round(float(df['yoy_pct'].mean()), 2),
            'latest': float(df['yoy_pct'].iloc[-1]),
        },
    }


if __name__ == '__main__':
    df = load_monthly_ppi()
    if df.empty:
        print('没有月度数据')
    else:
        print('\n=== 月度数据摘要 ===')
        summary = get_monthly_summary(df)
        for k, v in summary.items():
            print(f'{k}: {v}')
