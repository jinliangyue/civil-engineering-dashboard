"""
数据加载模块
功能：从 data/raw/ 目录读取所有 CSV 文件，按规范解析成 pandas DataFrame
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# 字段映射（用户下载数据可能用其他列名）
COLUMN_ALIASES = {
    'date': ['date', '日期', '时间', 'month', 'period'],
    'price': ['price', '价格', '单价', 'value', 'amount'],
    'material': ['material', '材料', '材料名', '材料名称', 'category'],
    'unit': ['unit', '单位'],
    'region': ['region', '地区', '城市', 'location'],
    'source': ['source', '来源', 'data_source'],
}

# 必填字段
REQUIRED_FIELDS = ['date', 'price', 'material']


def list_raw_files() -> List[Path]:
    """列出 data/raw/ 下所有 CSV 文件"""
    if not RAW_DATA_DIR.exists():
        logger.warning(f'原始数据目录不存在: {RAW_DATA_DIR}')
        return []
    return sorted(RAW_DATA_DIR.glob('*.csv'))


def parse_filename(filename: str) -> Dict[str, str]:
    """
    解析文件名 <材料名>_<地区代码>_<起始年月>_<结束年月>.csv
    示例：钢筋_北京_2020-01_2024-12.csv
    返回: {'material': '钢筋', 'region': '北京', 'start': '2020-01', 'end': '2024-12'}
    """
    stem = filename.replace('.csv', '').replace('.CSV', '')
    parts = stem.split('_')
    if len(parts) >= 4:
        return {
            'material': parts[0],
            'region': parts[1],
            'start': parts[2],
            'end': parts[3],
        }
    else:
        logger.warning(f'文件名格式不规范: {filename}')
        return {'material': 'unknown', 'region': 'unknown', 'start': '', 'end': ''}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名（用户下载的数据列名可能多样）
    内部统一为：date / price / material / unit / region / source
    """
    rename_map = {}
    for standard_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and standard_col not in df.columns:
                rename_map[alias] = standard_col
                break
    df = df.rename(columns=rename_map)
    return df


def validate_dataframe(df: pd.DataFrame, filename: str) -> tuple:
    """
    校验 DataFrame 是否符合规范
    返回: (是否通过, 错误信息列表)
    """
    errors = []
    # 检查必填字段
    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            errors.append(f'缺少必填字段: {field}')
    if errors:
        return False, errors
    # 检查数据行数
    if len(df) == 0:
        errors.append('数据为空')
        return False, errors
    # 检查日期格式
    try:
        pd.to_datetime(df['date'])
    except Exception as e:
        errors.append(f'日期格式无法解析: {e}')
    # 检查价格是否为数值
    try:
        df['price'].astype(float)
    except Exception as e:
        errors.append(f'价格列无法转为数值: {e}')
    if errors:
        return False, errors
    return True, []


def load_single_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """加载单个 CSV 文件，自动校验 + 标准化"""
    logger.info(f'加载文件: {filepath.name}')
    try:
        # 尝试不同编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            logger.error(f'无法解析文件编码: {filepath.name}')
            return None
        # 标准化列名
        df = normalize_columns(df)
        # 校验
        is_valid, errors = validate_dataframe(df, filepath.name)
        if not is_valid:
            logger.error(f'数据校验失败 {filepath.name}: {errors}')
            return None
        # 从文件名补充 region 和 material（如列里没有）
        meta = parse_filename(filepath.name)
        if 'material' not in df.columns:
            df['material'] = meta['material']
        if 'region' not in df.columns:
            df['region'] = meta['region']
        # 统一日期格式
        df['date'] = pd.to_datetime(df['date'])
        # 统一价格为 float
        df['price'] = df['price'].astype(float)
        logger.info(f'加载成功: {filepath.name} ({len(df)} 行)')
        return df
    except Exception as e:
        logger.error(f'加载文件出错 {filepath.name}: {e}')
        return None


def load_all_raw() -> pd.DataFrame:
    """加载 data/raw/ 下所有 CSV 文件，合并为一个 DataFrame"""
    files = list_raw_files()
    if not files:
        logger.warning('没有找到任何 CSV 文件')
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = load_single_csv(f)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f'合并完成: 总行数 {len(result)}, 材料种类 {result["material"].nunique()}')
    return result


def save_processed(df: pd.DataFrame, filename: str = 'all_cleaned.csv'):
    """保存处理后的数据到 data/processed/"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    logger.info(f'已保存: {filepath}')


def get_data_summary(df: pd.DataFrame) -> Dict:
    """生成数据摘要报告"""
    if df.empty:
        return {'status': 'empty'}
    return {
        'status': 'ok',
        'total_rows': len(df),
        'materials': df['material'].unique().tolist(),
        'regions': df['region'].unique().tolist(),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d'),
        },
        'price_range': {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean(),
        },
    }


if __name__ == '__main__':
    # 脚本入口：跑一次看效果
    df = load_all_raw()
    if not df.empty:
        summary = get_data_summary(df)
        print('\n=== 数据摘要 ===')
        for k, v in summary.items():
            print(f'{k}: {v}')