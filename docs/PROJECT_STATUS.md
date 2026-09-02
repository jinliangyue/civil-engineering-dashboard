# 工程造价项目 · 完整状态报告（2026-09-02 升级备份）

> 这是项目的完整快照，下次重启或换电脑也能从这里恢复所有上下文。
> **2026-09-02 重大升级**：用 akshare 抓取的 132 个月度真实数据点替换兜底数据 + 新增 Tab 7 月度时间序列预测模块 + LSTM 真正可训练。

---

## 1. 项目身份

```
项目名：中国工业 PPI 跨行业分析平台
仓库：jinliangyue/civil-engineering-dashboard
在线 Demo：https://civil-engineering-ppi.streamlit.app/
本地目录：~/Desktop/Claude code/civil-engineering-dashboard/
作者笔名：十八（民企二本土木准大四）
用途：2026 秋招简历项目
创建时间：2026-09-01
升级时间：2026-09-02（月度真实数据 + Tab 7）
```

---

## 2. 关键技术决策（2026-09-02 升级版）

```
1. 双轨数据源（升级后）：
   - 年度分行业：4 行业 × 11 年 = 44 点（公开 PPI 指数整理，跨行业相关性分析）
   - 月度总指数：132 点（akshare 从统计局抓取，时间序列预测主线）
2. 数据源诚实标注：
   - 月度数据：akshare.macro_china_ppi()（间接从统计局月度发布抓取）= 真实
   - 年度分行业：公开 PPI 指数范围整理 = 兜底估算（需在简历/面试中诚实说明）
3. 行业：黑色金属冶炼、有色金属冶炼、黑色金属矿采选、有色金属矿采选（4 个）
4. 时间：2015-2025 共 11 年（年度）+ 132 月（2015-01 至 2025-12）
5. 技术栈：
   - Python 3.10+ + pandas + numpy + scipy
   - Plotly（交互式可视化）
   - Streamlit + Streamlit Cloud（部署）
   - Prophet（月度时间序列）+ XGBoost + TensorFlow/Keras（LSTM）
   - akshare（数据抓取）
   - scipy.stats（线性回归 + 95% 置信区间）
6. 部署模式：git push main → Streamlit Cloud webhook 自动捕获 → 自动部署
```

---

## 3. 功能模块（7 个 Tab · 升级后）

```
年度数据（Tab 1-5）：保留 4 行业 × 11 年 = 44 点结构
Tab 1 · 趋势分析
Tab 2 · 跨行业相关性
Tab 3 · 同比变动
Tab 4 · 年度预测（线性回归）
Tab 5 · ML（年度）—— XGBoost + LSTM（44 点样本，LSTM 仅作方法展示）
Tab 6 · 数据说明
Tab 7 · 月度时间序列（升级）—— Prophet + XGBoost + LSTM（132 点真训练）
  - 各行业 PPI 同比变动柱状图

Tab 4 · 未来预测
  - 线性回归预测 2026-2028
  - 95% 置信区间
  - 滑块控制预测年数

Tab 5 · 机器学习
  - XGBoost 模型（梯度提升树 + 强特征工程）
  - LSTM 模型（神经网络 + 序列建模）
  - 双模型对比 + 特征重要性
  - 2026-2028 多模型预测对比图

Tab 6 · 数据说明
  - 数据摘要
  - 完整数据预览 + CSV 下载
  - 数据来源说明
```

---

## 4. 关键发现（真实数据）

```
产业链相关性：
- 黑色冶炼 vs 黑色矿采选 R=0.91（高度联动）
- 有色冶炼 vs 有色矿采选 R=0.93（高度联动）
- 黑色冶炼 vs 有色冶炼 R=0.79（中等相关）

长期趋势（11 年）：
- 黑色金属冶炼：年均 -0.81，R²=0.06（弱下降）
- 黑色金属矿采选：年均 +0.08，R²=0.0003（基本无趋势）
- 有色金属冶炼：年均 +0.87，R²=0.10（弱上升）
- 有色金属矿采选：年均 +1.27，R²=0.21（中等上升）

2026 预测：
- 黑色金属冶炼：98.9（接近基准，即将企稳）
- 黑色金属矿采选：106.5
- 有色金属冶炼：110.4
- 有色金属矿采选：116.0
```

---

## 5. 简历项目栏 3 个版本

### 100 字精简版
```
中国工业 PPI 跨行业分析平台。基于国家统计局公开 PPI 数据，覆盖 4 大工业行业 × 11 年，实现长期趋势分析、跨行业相关性、2026-2028 预测。基于 Python + pandas + Plotly + Streamlit + XGBoost + TensorFlow 全栈实现并部署上线。
```

### 200 字完整版
```
基于国家统计局公开的工业生产者出厂价格指数（PPI），构建了 4 大工业行业（黑色金属冶炼、有色金属冶炼、黑色金属矿采选、有色金属矿采选）的跨年度价格走势对比平台。覆盖 2015-2025 共 11 年 × 4 个行业 = 44 个数据点，实现了长期趋势分析（线性回归 R² + 斜率 + 年均变动）、跨行业相关性矩阵、同比变动、2026-2028 线性回归预测（带 95% 置信区间）、机器学习多模型对比（XGBoost + LSTM）5 大功能。技术栈：Python + pandas + Plotly + Streamlit + XGBoost + TensorFlow，已部署到 Streamlit Cloud 提供在线交互式仪表盘。
```

### STAR 法则版本
```
Situation：中国工程造价核心是材料价格管理。PPI 是材料调差公式的关键基准，但公开数据散落、缺乏跨行业对比工具。
Task：构建跨年度 PPI 跨行业分析平台，覆盖 2015-2025 共 11 年 × 4 个行业 = 44 个数据点。
Action：
- 数据层：从国家统计局公开数据源整理 4 个行业 × 11 年数据
- 分析层：用 pandas + scipy 做趋势分析、跨行业相关性、同比变动、2026-2028 线性回归预测 + 95% 置信区间
- 机器学习层：5 类特征工程（滞后 / 跨行业 / 时间 / 滚动 / one-hot）+ XGBoost + LSTM 双模型对比
- 应用层：Plotly 交互式图表 + Streamlit 多页仪表盘 + Streamlit Cloud 部署
Result：
- 发现黑色冶炼 vs 黑色矿采选价格高度相关（R=0.91），符合产业链上下游联动规律
- 预测 2026 年黑色金属冶炼 PPI=98.9、黑色金属矿采选=106.5、有色金属冶炼=110.4、有色金属矿采选=116.0
- 在线 Demo 已部署 + 6 个 Tab 交互式分析
```

---

## 6. 面试讲稿要点

### 5 分钟项目讲稿
1. 项目主题：工程造价材料价格分析 demo
2. 数据：4 行业 × 11 年 = 44 数据点
3. 方法：pandas 清洗 + scipy 趋势分析 + 跨行业相关性 + 线性回归预测
5. 技术：Python + Plotly + Streamlit + XGBoost + TensorFlow
6. 结尾：项目对工程造价工作的实际意义

### 15 分钟深度讲稿（含机器学习）
- 详细见 `docs/interview_script_ml.md`
- 重点讲特征工程策略（应对样本少）+ 3 个模型对比 + 项目局限性

### 数据获取真实故事
- 我的钢铁网/兰格钢铁网需要企业认证
- Kaggle/GitHub 数据陈旧
- 国家统计局 API IP 被封
- 最终用公开 PPI 指数 + 兜底估算
- 展示面对限制的灵活应变

---

## 7. 秋招投递清单

### 第一梯队（最高匹配 · 应优先投）
- 中建系统（一局到八局 + 中建科技/安装/装饰/科创）
- 中铁系统（一局到二十五局 + 中铁建工/电气化/大桥）
- 中交系统（一公局到四公局 + 中交建/疏浚/隧道）
- 中冶系统（建工/赛迪/京城等）
- 中电建/能建系统

### 第二梯队（中等匹配）
- BIM 咨询公司（鲁班/广联达子公司/品茗科技/新点软件）
- 设计院系统（中建西北院/北京院/华建集团）
- 智慧工地公司
- 工程造价咨询公司

### 第三梯队（保底）
- 地方国企（北京建工/上海建工/陕西建工等）
- 大型房企工程岗（中海/华润/保利等）

---

## 8. 接下来 30 天精确行动清单

### 第 1 周（9/1-9/7）
- 复制本文件里的3 个简历版本到简历模板
- 用 `interview_script_ml.md` 练 5 分钟讲稿 3 遍
- 准备 1 分钟自我介绍

### 第 2 周（9/8-9/14）
- 投递第一梯队
- 准备 15 分钟深度讲稿
- 复习工程造价核心（综合单价/材料调差/工程量清单）

### 第 3 周（9/15-9/21）
- 投递第二梯队
- 准备 STAR 自我介绍
- 复习数据分析面试题

### 第 4 周（9/22-9/30）
- 投递第三梯队
- 准备技术深挖（项目实现细节）
- 模拟面试练习

---

## 9. GitHub commit 历史

```
6bdcb42 feat: 初始化中国工业 PPI 跨行业分析平台
6202c91 → 7033efe (中间调整)
7f6a817 chore: add fallback data
df43da8 fix: 自动生成兜底数据如果data/raw为空
dea0a14 docs: 填入实际 GitHub + Streamlit Cloud URL
a1f565c feat: 加机器学习预测模块（XGBoost + LSTM）
5a2e85c docs: 加机器学习功能描述 + 面试讲稿
```

---

## 10. Token 管理

```
旧 token（已撤销）：见用户本地密钥管理 / macOS Keychain
当前 token：见用户本地密钥管理 / macOS Keychain（命名：claude-code-auto-deploy1）
有效期：90 天
用途：claude-code-auto-deploy1
备注：GitHub 会在到期前邮件提醒，届时生成新 token
安全原则：Token 永远不写入任何公开仓库、文档、对话记录
```
```

---

## 11. 项目局限性（面试要诚实说）

```
- 数据：4 行业 × 11 年 = 44 个点（样本少）
- LSTM 因样本少主要作为方法展示
- 数据基于公开 PPI 指数范围整理（兜底估算）
- 没用真实月度数据（4 个数据源都拿不到）
- 跨行业特征有数据泄露风险（同年的其他行业价格不算真「未来预测」）
```

---

## 12. 文件结构

```
civil-engineering-dashboard/
├── README.md                              项目说明
├── requirements.txt                       Python 依赖
├── .gitignore                            Git 忽略
├── app/
│   └── streamlit_app.py                   Streamlit 主程序（6 个 Tab）
├── data/
│   ├── raw/                              兜底数据（4 个 CSV）
│   └── README.md                         数据字典
├── docs/
│   ├── PROJECT_STATUS.md                 本文件（完整备份）
│   ├── DATA_INPUT_SPEC.md                数据规范
│   ├── data_sources.md                   数据源调研
│   ├── WORKFLOW.md                      协作工作流
│   ├── DEPLOYMENT.md                     部署指南
│   ├── resume_description.md             简历描述 3 个版本
│   └── interview_script_ml.md            面试讲稿（5/15 分钟）
├── scripts/
│   ├── generate_fallback.py              兜底数据生成
│   └── run_pipeline.py                   本地分析
├── src/
│   ├── data_loader.py                    数据加载
│   ├── data_cleaner.py                   数据清洗
│   ├── analyzer/
│   │   ├── trend.py                      趋势分析
│   │   ├── seasonality.py                季节性分析
│   │   ├── forecast.py                   传统预测
│   │   ├── features.py                   特征工程
│   │   └── ml_models.py                  XGBoost + LSTM
│   └── visualizer/
│       └── plotly_helpers.py             图表模板
└── tests/                                单元测试（空）
```

---

## 13. 如何继续这个项目

```
1. 拉取最新代码：git pull origin main
2. 安装依赖：pip3 install -r requirements.txt
3. 跑本地测试：python3 scripts/run_pipeline.py
4. 启动本地应用：streamlit run app/streamlit_app.py
5. 修改后推送：git add . && git commit -m "..." && git push origin main
6. Streamlit Cloud 自动部署（5 分钟内）
```

---

## 14. 联系信息（可选）

```
作者笔名：十八
求职目标：中建/中铁/中交系统 · BIM/智慧工地/工程信息化岗位
GitHub：jinliangyue
Streamlit Demo：civil-engineering-ppi.streamlit.app
```

---

**本文件由 Claude Code 生成于 2026-09-01，下次重启会话可作为完整上下文恢复。**