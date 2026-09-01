# 数据目录说明

## 目录结构

```
data/
├── raw/                          # 原始数据（用户下载的 CSV）
│   └── <材料>_<地区>_<起始>_<结束>.csv
├── processed/                    # 清洗后的数据
│   ├── all_cleaned.csv
│   └── figures/                  # 生成的图表
└── README.md                     # 本文件
```

## 当前数据状态

**状态：等待用户下载数据**

`data/raw/` 目录为空，请按以下步骤操作：

1. 按 `docs/data_sources.md` 调研数据源
2. 按 `docs/DATA_INPUT_SPEC.md` 下载 + 整理数据
3. 把 CSV 文件放进 `data/raw/`

## 字段字典

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| date | date | ✅ | 日期，YYYY-MM-DD 格式 |
| price | float | ✅ | 价格，正数 |
| material | str | ✅ | 材料名（与文件名一致） |
| unit | str | 推荐 | 单位（元/吨/立方米/平方米） |
| region | str | 推荐 | 地区 |
| source | str | 推荐 | 数据来源 URL |
| is_outlier | int | 自动 | 异常值标记（0/1） |

## 数据流

```
用户下载 CSV
  ↓
data/raw/                    (用户操作)
  ↓
src/data_loader.py           (Claude Code 处理)
  ↓
src/data_cleaner.py          (清洗 + 标准化)
  ↓
data/processed/all_cleaned.csv
  ↓
src/analyzer/trend.py        (趋势分析)
src/analyzer/seasonality.py  (季节性分析)
src/analyzer/forecast.py     (预测)
  ↓
data/processed/figures/      (图表输出)
  ↓
app/streamlit_app.py         (Streamlit 仪表盘)
  ↓
Streamlit Cloud              (部署上线)
```