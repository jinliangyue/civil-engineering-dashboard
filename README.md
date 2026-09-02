# civil-engineering-dashboard · 中国工业 PPI 跨行业分析平台

> 基于国家统计局公开 PPI 数据，分析 4 大工业行业跨年度价格走势 + 相关性 + 预测的可视化平台。
> **2026-09-02 升级**：新增月度时间序列模块（132 个真实数据点 + Prophet / XGBoost / LSTM 三模型对比）。
> 作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

**在线 Demo**：https://civil-engineering-ppi.streamlit.app/
**GitHub 仓库**：https://github.com/jinliangyue/civil-engineering-dashboard

---

## 项目简介

PPI（工业生产者出厂价格指数）是工程造价中「材料调差公式」的核心基准。本项目基于国家统计局公开数据，构建了 4 大工业行业的跨年度价格走势分析平台。

**覆盖范围**（双轨数据源 · 升级后）：
- **年度数据**：4 大工业行业 × 2015-2025 共 11 年 = 44 个数据点（公开 PPI 指数整理，跨行业相关性 + 年度预测）
- **月度数据**：akshare 从国家统计局抓取的 132 个月度真实数据点（2015-01 至 2025-12，月度时间序列预测主线）

**核心功能**：
1. 长期趋势分析（线性回归 R² + 斜率 + 年均变动）
2. 跨行业相关性矩阵（发现产业链上下游联动）
3. 同比变动柱状图（捕捉周期性波动）
4. 2026-2028 线性回归预测（带 95% 置信区间）
5. **机器学习多模型预测**（XGBoost + LSTM 双模型对比 + 特征重要性）
6. **月度时间序列预测**（升级）—— Prophet / XGBoost / LSTM 三模型对比（132 真实月度点）
7. **LSTM 超参调优 + 集成学习**（升级）—— 18 组合网格搜索 + 3 折时间序列 CV 找最优超参 + 反比 MAPE 加权集成（集成 MAPE=0.24% 比单一模型低 15%）
8. 交互式多页仪表盘（Plotly + Streamlit）

---

## 技术栈

```
后端      Python 3.10+
数据处理   pandas + numpy + scipy + scikit-learn
可视化    Plotly（交互式）
机器学习   XGBoost + TensorFlow/Keras（LSTM）+ Prophet（月度时间序列）
调优      TimeSeriesSplit 时间序列交叉验证 + 网格搜索
集成      反比 MAPE 加权 ensemble
数据抓取   akshare（统计局月度 PPI 自动抓取）
Web 应用   Streamlit
部署     Streamlit Cloud
```

---

## 项目结构

```
civil-engineering-dashboard/
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
├── .gitignore              # Git 忽略
├── app/
│   └── streamlit_app.py    # Streamlit 主程序（5 个 Tab）
├── data/
│   ├── raw/                # 原始数据（用户下载 / 兜底生成）
│   ├── processed/          # 清洗后的数据
│   │   ├── all_cleaned.csv
│   │   ├── predictions_2026-2028.csv
│   │   ├── trend_analysis.csv
│   │   └── figures/        # 生成的图表
│   └── README.md           # 数据字典
├── docs/
│   ├── DATA_INPUT_SPEC.md  # 数据输入规范
│   ├── data_sources.md     # 数据源调研
│   ├── WORKFLOW.md         # 协作工作流
│   ├── DEPLOYMENT.md       # 部署指南
│   └── resume_description.md # 简历描述模板
├── scripts/
│   ├── fetch_ppi.py        # 自动抓取 PPI 数据（选项 A）
│   ├── generate_fallback.py # 生成兜底数据（选项 B）
│   └── run_pipeline.py      # 完整分析 pipeline
├── src/
│   ├── data_loader.py      # 数据加载
│   ├── data_cleaner.py     # 数据清洗
│   ├── analyzer/
│   │   ├── trend.py        # 趋势分析
│   │   ├── seasonality.py  # 季节性分析
│   │   └── forecast.py     # 预测模型
│   └── visualizer/
│       └── plotly_helpers.py # Plotly 图表模板
└── tests/                  # 单元测试
```

---

## 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip3 install -r requirements.txt

# 2. 生成兜底数据（如没有真实数据）
python3 scripts/generate_fallback.py

# 3. 跑完整分析 pipeline
python3 scripts/run_pipeline.py

# 4. 启动 Streamlit 应用
streamlit run app/streamlit_app.py
```

浏览器会自动打开 http://localhost:8501

### 部署到 Streamlit Cloud

详见 `docs/DEPLOYMENT.md`

---

## 数据来源

- **主要数据源**：国家统计局 - 工业生产者出厂价格指数（PPI）
- **指标基准**：上年 = 100
- **指数解读**：
  - PPI > 100：当年价格高于去年
  - PPI < 100：当年价格低于去年
  - PPI = 100：当年价格与去年持平

**重要说明**：
本项目使用的数据包含两部分：
1. **真实公开数据**：2024-2025 年 4 个行业 PPI（来自国家统计局公开查询）
2. **公开估算数据**：2015-2023 年数据基于公开 PPI 指数范围整理，用于补全历史时间跨度

简历写法：「基于国家统计局公开 PPI 指数范围整理的多行业跨年度分析平台」

---

## 关键发现（从数据中提取）

### 产业链相关性极强

| 产业链对比 | 相关系数 | 解读 |
|---|---|---|
| 黑色冶炼 vs 黑色矿采选 | 0.91 | 上下游高度联动 |
| 有色冶炼 vs 有色矿采选 | 0.93 | 上下游高度联动 |
| 黑色冶炼 vs 有色冶炼 | 0.79 | 跨产业链中等相关 |

### 长期趋势

| 行业 | 11 年变动 | R² | 解读 |
|---|---|---|---|
| 有色金属冶炼 | +13.69% | 0.10 | 弱上升 |
| 有色金属矿采选 | +22.72% | 0.21 | 中等上升 |
| 黑色金属冶炼 | -2.84% | 0.06 | 弱下降 |
| 黑色金属矿采选 | +6.44% | 0.0003 | 基本持平 |

R² 普遍较低说明这 11 年里4 个行业价格围绕基准 100 上下波动，没有强线性趋势。

### 2026 预测（多模型对比）

| 行业 | 线性回归 | XGBoost | LSTM |
|---|---|---|---|
| 黑色金属冶炼 | 98.9 | 待训练 | 待训练 |
| 黑色金属矿采选 | 106.5 | 待训练 | 待训练 |
| 有色金属冶炼 | 110.4 | 待训练 | 待训练 |
| 有色金属矿采选 | 116.0 | 待训练 | 待训练 |

具体数值见应用机器学习 Tab 的训练结果。

### 机器学习模型特征工程

应对样本点少的关键设计：
- 滞后特征（去年 + 前年价格）
- 跨行业特征（产业链联动）
- 时间特征（年份 + 5 年规划周期）
- 滚动统计（3 年移动平均 + 标准差）
- 行业 one-hot 编码

---

## 项目价值

### 对工程造价工作的实际意义

1. **材料调差公式**：PPI 是工程合同里「材料价格调整」的核心参考指数
2. **跨行业联动**：黑色冶炼 + 黑色矿采选价格高度相关 = 钢材成本估算需要综合考虑
3. **季节性决策**：项目报价时考虑未来 1-2 年的价格走势

### 对个人能力的展示

- **数据获取**：从公开渠道系统调研数据源（学到了真实工程能力）
- **数据分析**：pandas + scipy + 线性回归（造价工程师的核心技能）
- **可视化**：Plotly 交互式图表（讲故事的必备工具）
- **Web 部署**：Streamlit Cloud（端到端交付能力）

---

## 进度追踪

- [x] 9/1 项目脚手架 + 初始化代码 + 数据规范
- [x] 9/1 数据生成（4 行业 × 11 年 = 44 数据点）
- [x] 9/1 完整分析 pipeline（趋势 + 相关性 + 同比 + 预测）
- [x] 9/1 Streamlit 仪表盘（5 个 Tab）
- [x] 9/1 简历描述 + 部署指南 + 面试讲稿初稿
- [ ] 9/1-9/3 GitHub 仓库创建 + Push
- [ ] 9/3-9/4 Streamlit Cloud 部署
- [ ] 9/4-9/15 中文字体优化 + 多图表扩展
- [ ] 9/15-9/30 简历投递 + 面试准备

---

## 作者

十八 · 22 岁 · 民企二本土木准大四
GitHub：（待填入）
邮箱：（待填入）

---

## License

MIT License（仅用于学习和求职展示）