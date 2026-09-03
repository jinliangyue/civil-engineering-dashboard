"""
PPI 数据自动抓取脚本（选项 A · 历史版本）
从 akshare 间接拉取国家统计局月度 PPI 数据

> 当前正式数据获取方式：直接调用 `src.ppi_monthly.load_monthly_ppi()`，
> 该函数内部使用 akshare.macro_china_ppi() 获取 132 个月度点。
> 此独立脚本仅保留作为历史参考，不再是项目主流程的一部分。

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

使用方法：
1. 打开终端（Terminal）
2. 切换到项目根目录（任意路径）
3. 安装依赖：pip install requests pandas
4. 运行脚本：python3 scripts/fetch_ppi.py
5. 等待脚本跑完，会自动生成 CSV 文件到 data/raw/
"""

import requests
import pandas as pd
import time
import os
from pathlib import Path

# 国家统计局数据接口
API_URL = 'https://data.stats.gov.cn/easyquery.htm'

# 4 个行业 PPI 指标编码
INDICATORS = {
    '黑色金属冶炼和压延加工业': 'A090101',
    '有色金属冶炼和压延加工业': 'A090201',
    '黑色金属矿采选业': 'A090103',
    '有色金属矿采选业': 'A090203',
}


def fetch_ppi_data(indicator_code: str, start_year: int, end_year: int):
    """
    抓取指定指标的 PPI 数据
    indicator_code: 行业指标编码
    start_year / end_year: 起止年份
    """
    params = {
        'm': 'QueryData',
        'dbcode': 'hgjd',
        'rowcode': 'zb',
        'colcode': 'sj',
        'wds': '[]',
        'dfwds': f'[{{"wdcode":"zb","valuecode":"{indicator_code}"}},{{"wdcode":"sj","valuecode":"{start_year}-{end_year}"}}]',
    }
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'  抓取失败: {e}')
        return None


def parse_ppi_response(data: dict, material: str):
    """
    解析返回的 JSON 数据
    """
    rows = []
    if not data or 'returndata' not in data:
        return rows
    datanodes = data['returndata'].get('datanodes', [])
    for node in datanodes:
        try:
            value_data = node.get('data', {})
            value = value_data.get('data')
            wds = node.get('wds', [])
            if len(wds) < 2:
                continue
            year_code = wds[1].get('valuecode', '')
            if value is None or value == '' or value == 0:
                continue
            rows.append({
                'date': f'{year_code}-12-31',
                'price': float(value),
                'material': material,
                'unit': '指数(上年=100)',
                'region': '全国',
                'source': '国家统计局',
            })
        except (KeyError, ValueError, IndexError) as e:
            continue
    return rows


def main():
    """主函数"""
    print('=' * 60)
    print('PPI 数据自动抓取脚本')
    print('=' * 60)
    # 项目根目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for material, code in INDICATORS.items():
        print(f'\n抓取: {material}...')
        data = fetch_ppi_data(code, 2015, 2025)
        if data:
            rows = parse_ppi_response(data, material)
            all_rows.extend(rows)
            print(f'  获取 {len(rows)} 条记录')
        else:
            print(f'  无数据返回')
        time.sleep(1.5)
    if not all_rows:
        print('\n未获取到任何数据')
        print('可能原因：')
        print('  1. 国家统计局 API 参数已变更')
        print('  2. 网络无法访问 data.stats.gov.cn')
        print('  3. 指标编码不正确')
        print('\n解决方案：')
        print('  1. 检查网络是否正常（浏览器打开 data.stats.gov.cn 试试）')
        print('  2. 使用项目主流程：from src.ppi_monthly import load_monthly_ppi')
        print('     （通过 akshare.macro_china_ppi() 获取 132 个月度真实点）')
        return
    # 保存
    df = pd.DataFrame(all_rows)
    df = df.sort_values(['material', 'date']).reset_index(drop=True)
    for material, group in df.groupby('material'):
        years = sorted(group['date'].str[:4].unique())
        if not years:
            continue
        start_year = years[0]
        end_year = years[-1]
        # 文件名简化（处理特殊字符）
        material_safe = material.replace('和压延加工业', '').replace('矿采选业', '')
        filename = f'{material_safe}_全国_{start_year}-{end_year}.csv'
        filepath = output_dir / filename
        group.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f'\n已保存: {filepath.name} ({len(group)} 行)')
    # 汇总
    summary = df.groupby('material').agg(
        count=('price', 'count'),
        min_year=('date', 'min'),
        max_year=('date', 'max'),
        mean_price=('price', 'mean'),
    ).round(2)
    print('\n' + '=' * 60)
    print('数据摘要')
    print('=' * 60)
    print(summary)
    print(f'\n总计: {len(df)} 条记录, 4 个行业, {df["date"].min()[:4]} 到 {df["date"].max()[:4]}')


if __name__ == '__main__':
    main()