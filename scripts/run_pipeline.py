"""
PPI 数据完整分析 pipeline
适配年度数据：4 个行业 × 11 年（2015-2025）
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

使用方法：
1. cd ~/Desktop/Claude\ code/civil-engineering-dashboard
2. python3 scripts/run_pipeline.py
3. 等待完成，会输出图表到 data/processed/figures/ + 摘要到终端
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非 GUI 模式
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.data_loader import load_all_raw, get_data_summary, save_processed


# ============== 1. 数据加载 ==============
print('=' * 60)
print('第 1 步：加载数据')
print('=' * 60)
df_raw = load_all_raw()
if df_raw.empty:
    print('错误：data/raw/ 下没有找到 CSV 文件')
    sys.exit(1)
summary = get_data_summary(df_raw)
print(f'加载成功: {summary["total_rows"]} 行')
print(f'  材料: {summary["materials"]}')
print(f'  地区: {summary["regions"]}')
print(f'  时间: {summary["date_range"]["start"]} ~ {summary["date_range"]["end"]}')


# ============== 2. 数据清洗 ==============
print('\n' + '=' * 60)
print('第 2 步：数据清洗')
print('=' * 60)
# 数据本身已经是干净的，只需排序 + 异常值检测
df_clean = df_raw.copy()
df_clean = df_clean.sort_values(['material', 'date']).reset_index(drop=True)
# 检查缺失
missing = df_clean['price'].isnull().sum()
print(f'缺失值: {missing}')
# 检查异常（用 IQR）
outliers_count = 0
for material, group in df_clean.groupby('material'):
    q1 = group['price'].quantile(0.25)
    q3 = group['price'].quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((group['price'] < q1 - 1.5 * iqr) | (group['price'] > q3 + 1.5 * iqr)).sum()
    if n_outliers > 0:
        print(f'  {material}: {n_outliers} 个异常值')
    outliers_count += n_outliers
print(f'总异常值: {outliers_count}')
# 保存清洗后数据
save_processed(df_clean, 'all_cleaned.csv')
print('已保存: data/processed/all_cleaned.csv')


# ============== 3. 趋势分析 ==============
print('\n' + '=' * 60)
print('第 3 步：趋势分析')
print('=' * 60)
trend_results = []
for material, group in df_clean.groupby('material'):
    group = group.sort_values('date')
    years = pd.to_datetime(group['date']).dt.year.values
    prices = group['price'].values
    # 线性回归（年份 vs 价格）
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, prices)
    # 同比变动
    yoy = group['price'].pct_change() * 100
    trend_results.append({
        '行业': material,
        '起始年': int(years[0]),
        '结束年': int(years[-1]),
        '起始值': round(prices[0], 2),
        '结束值': round(prices[-1], 2),
        '斜率(每年)': round(slope, 4),
        'R²': round(r_value ** 2, 4),
        '年均变动': round(slope, 2),
        '总变动 %': round((prices[-1] / prices[0] - 1) * 100, 2),
    })
df_trend = pd.DataFrame(trend_results)
print(df_trend.to_string(index=False))


# ============== 4. 跨行业相关性 ==============
print('\n' + '=' * 60)
print('第 4 步：跨行业相关性')
print('=' * 60)
df_pivot = df_clean.pivot_table(values='price', index='date', columns='material')
corr = df_pivot.corr()
print(corr.round(4).to_string())


# ============== 5. 简单线性回归预测 ==============
print('\n' + '=' * 60)
print('第 5 步：2026-2028 简单线性回归预测')
print('=' * 60)
predictions = []
for material, group in df_clean.groupby('material'):
    group = group.sort_values('date')
    years = pd.to_datetime(group['date']).dt.year.values
    prices = group['price'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, prices)
    # 预测 2026-2028
    for future_year in [2026, 2027, 2028]:
        pred = intercept + slope * future_year
        # 置信区间（95%）
        n = len(years)
        t_val = stats.t.ppf(0.975, n - 2)
        se = std_err * np.sqrt(1 + 1/n + (future_year - years.mean()) ** 2 / np.sum((years - years.mean()) ** 2))
        predictions.append({
            '行业': material,
            '预测年份': future_year,
            '预测 PPI': round(pred, 2),
            '下界 95%': round(pred - t_val * se, 2),
            '上界 95%': round(pred + t_val * se, 2),
            'R²': round(r_value ** 2, 4),
        })
df_pred = pd.DataFrame(predictions)
print(df_pred.to_string(index=False))


# ============== 6. 生成图表 ==============
print('\n' + '=' * 60)
print('第 6 步：生成图表')
print('=' * 60)
figures_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)

# 图 1：4 行业趋势线对比
fig, ax = plt.subplots(figsize=(12, 6))
df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
colors = {'黑色金属冶炼': '#3b82f6', '有色金属冶炼': '#ef4444', '黑色金属矿采选': '#10b981', '有色金属矿采选': '#f59e0b'}
for material, group in df_clean.groupby('material'):
    group_sorted = group.sort_values('year')
    ax.plot(group_sorted['year'], group_sorted['price'], marker='o', linewidth=2, label=material, color=colors.get(material, '#64748b'))
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='基准线 100')
ax.set_title('中国 4 大工业行业 PPI 跨年度走势（2015-2025）', fontsize=14, fontweight='bold')
ax.set_xlabel('年份')
ax.set_ylabel('PPI 指数（上年=100）')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'fig1_trend.png', dpi=150, bbox_inches='tight')
print(f'已生成: fig1_trend.png')
plt.close()

# 图 2：跨行业相关性热图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', color='black')
ax.set_title('4 大工业行业 PPI 相关性热图', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='相关系数')
plt.tight_layout()
plt.savefig(figures_dir / 'fig2_correlation.png', dpi=150, bbox_inches='tight')
print(f'已生成: fig2_correlation.png')
plt.close()

# 图 3：同比变动柱状图
fig, ax = plt.subplots(figsize=(12, 6))
df_clean['yoy'] = df_clean.groupby('material')['price'].pct_change() * 100
materials = df_clean['material'].unique()
width = 0.2
x = np.arange(len(df_clean['year'].unique()))
years = sorted(df_clean['year'].unique())
for i, material in enumerate(materials):
    group = df_clean[df_clean['material'] == material].sort_values('year')
    yoy = group['yoy'].fillna(0).values
    ax.bar(x + i * width, yoy, width, label=material, color=colors.get(material, '#64748b'))
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(years, rotation=45)
ax.set_title('各行业 PPI 同比变动（%）', fontsize=14, fontweight='bold')
ax.set_xlabel('年份')
ax.set_ylabel('同比变动 %')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(figures_dir / 'fig3_yoy.png', dpi=150, bbox_inches='tight')
print(f'已生成: fig3_yoy.png')
plt.close()

# 图 4：预测可视化
fig, ax = plt.subplots(figsize=(12, 6))
for material in df_clean['material'].unique():
    group = df_clean[df_clean['material'] == material].sort_values('year')
    # 历史
    ax.plot(group['year'], group['price'], marker='o', linewidth=2, label=f'{material} (历史)', color=colors.get(material, '#64748b'))
    # 预测
    pred_group = df_pred[df_pred['行业'] == material]
    ax.plot(pred_group['预测年份'], pred_group['预测 PPI'], marker='s', linestyle='--', linewidth=2, label=f'{material} (预测)', color=colors.get(material, '#64748b'), alpha=0.6)
ax.set_title('2026-2028 PPI 预测（线性回归）', fontsize=14, fontweight='bold')
ax.set_xlabel('年份')
ax.set_ylabel('PPI 指数（上年=100）')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvline(x=2025.5, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(figures_dir / 'fig4_forecast.png', dpi=150, bbox_inches='tight')
print(f'已生成: fig4_forecast.png')
plt.close()

# 保存预测结果
df_pred.to_csv(figures_dir.parent / 'predictions_2026-2028.csv', index=False, encoding='utf-8-sig')
df_trend.to_csv(figures_dir.parent / 'trend_analysis.csv', index=False, encoding='utf-8-sig')

print('\n' + '=' * 60)
print('完成')
print('=' * 60)
print(f'图表位置: {figures_dir}')
print(f'预测文件: data/processed/predictions_2026-2028.csv')
print(f'趋势分析: data/processed/trend_analysis.csv')