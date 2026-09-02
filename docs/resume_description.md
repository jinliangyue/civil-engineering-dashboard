# 简历项目描述（P0.8 重构 · 基于 P0.5 / P0.6 真实实验结果）

> 本文件提供 3 个版本的简历项目描述。所有数字均来自严格的实验审计（P0.5 Final Test + P0.6 Walk-forward），不允许使用任何旧的 fallback / leakage 相关数字。

---

## 项目名

**中国工业 PPI 月度时间序列分析与预测平台**

---

## 100 字精简版（项目栏短描述）

> 基于国家统计局公开的工业生产者出厂价格指数（PPI），构建月度时间序列分析与预测平台。覆盖 2015-01 ~ 2025-12 共 132 个真实月度观测，集成 4 类统计基准、Prophet、XGBoost 与 PyTorch LSTM，并实现 Validation-weighted Ensemble 与 Walk-forward Validation 稳健性评估。

---

## 200 字完整版（项目栏完整描述）

> 基于 akshare 抓取的国家统计局月度 PPI 数据（2015-01 ~ 2025-12，132 个真实月度观测），构建中国工业 PPI 月度时间序列分析与预测平台。采用严格的 Train/Validation/Test 划分（84 / 24 / 24 月），最终 Test 区间 2024-01 ~ 2025-12 共 24 个月严格 OOS 评估。评估 7 个模型（Naive / Seasonal Naive / MA / SES / Prophet / XGBoost / LSTM），其中 XGBoost Test MAPE=0.36%、LSTM MAPE=0.44%、Validation-weighted Ensemble MAPE=0.36% / R²=0.57。同时实现 3-fold expanding-window Walk-forward Validation（2021/2022/2023 各 12 个月）验证多年稳健性：XGBoost Mean MAPE=1.60%、LSTM Mean MAPE=1.41%。完整 leakage 防控：LSTM scaler 仅 fit 训练段、XGBoost 因果特征 + 子进程隔离、所有 rolling/lag/yoy/mom 特征严格 causal。技术栈：Python · pandas · Prophet · XGBoost · PyTorch LSTM · akshare · Streamlit，已部署到 Streamlit Cloud。

---

## STAR 法则版本（面试讲稿用）

**Situation（背景）**

中国工程造价的核心是材料价格管理，而 PPI（工业生产者出厂价格指数）是材料调差公式的关键基准。公开 PPI 月度数据散落在国家统计局各类公报中，缺乏跨年度对比工具和严谨的预测流程。

**Task（任务）**

构建一个端到端的月度 PPI 时序预测与评估平台，覆盖完整实验边界、数据隔离、leakage 防控和模型对比。

**Action（行动）**

- **数据层**：通过 akshare 从国家统计局抓取 2015-01 ~ 2025-12 共 132 个月度 PPI 真实数据点
- **特征工程**：15 个手工因果特征（lag 1/3/6/12 + rolling mean/std 3/6/12 + 同比/环比），全部基于 shift(1) 保证因果性
- **模型层**：评估 7 个模型（Naive / Seasonal Naive / MA / SES / Prophet / XGBoost / LSTM），其中 LSTM 使用 P0.3 锁定的网格搜索参数（hidden_size=32 / dropout=0.1 / seq_length=6）
- **集成层**：Validation 反比 MAPE 加权，权重在 Final Test 评估前锁定（Test 不参与权重计算）
- **稳健性层**：3-fold expanding-window Walk-forward Validation 验证模型在不同波动阶段的稳定性
- **Leakage 防控层**：LSTM scaler 仅 fit 训练段、XGBoost 因果特征、rolling one-step-ahead、Test 集只用一次

**Result（结果）**

- **P0.5 Final Test（2024-01 ~ 2025-12，24 月 OOS）**：
  - XGBoost MAPE = 0.3558%
  - LSTM MAPE = 0.4387%
  - Validation-weighted Ensemble MAPE = 0.3551%，R² = 0.5664
- **P0.6 Walk-forward（2021 / 2022 / 2023 各 12 月）**：
  - Naive Mean MAPE = 1.02%（基准）
  - XGBoost Mean MAPE = 1.60%
  - LSTM Mean MAPE = 1.41%

完整实验文档包含 9 项 metrics 单元测试、4 项 leakage checks、subprocess isolation 工程方案。

---

## 简历项目栏（标准格式）

```
2026.09 - 2026.09    中国工业 PPI 月度时间序列分析与预测平台
                    技术栈：Python · pandas · Plotly · Streamlit · Prophet · XGBoost · PyTorch · akshare

● 基于 akshare 抓取国家统计局月度 PPI 数据 132 点（2015-01 ~ 2025-12）
● 7 模型对比：4 个统计基准 + Prophet + XGBoost + PyTorch LSTM
● Final Test（2024-01 ~ 2025-12，24 月 OOS）：Ensemble MAPE 0.36%、R² 0.57
● Walk-forward Validation（2021 / 2022 / 2023）：XGBoost Mean MAPE 1.60%
● 严格 leakage 防控：因果特征 + scaler 隔离 + rolling 预测 + 子进程隔离
● Streamlit Cloud 部署：civil-engineering-ppi.streamlit.app
● GitHub：jinliangyue/civil-engineering-dashboard
```

---

## 面试讲稿（5 分钟版）

**开场（30 秒）**

这是 PPI 月度时间序列预测平台，132 个真实月度数据点，7 个模型对比，严格 OOS 评估。

**数据（1 分钟）**

通过 akshare 从国家统计局抓取 132 个月度 PPI 数据（2015-01 ~ 2025-12）。这是真实数据，不是模拟，也不是手工估算。我做数据获取时遇到 4 个数据源都失败（我的钢铁需要企业认证、Kaggle 数据陈旧、统计局 API IP 被封），最终用 akshare 间接抓取成功——展示面对限制的应变能力。

**方法（2 分钟）**

严格 Train/Validation/Test 划分 84/24/24。Final Test 区间 2024-2025，24 个月严格 OOS，Test 只能用于最终评估。

评估 7 个模型：4 个统计基准（Naive、Seasonal Naive、MA、SES）、Prophet、XGBoost、PyTorch LSTM。XGBoost 用 15 个手工因果特征（lag + rolling + 同比环比），全部 shift(1) 保证因果性。LSTM 用 P0.3 锁定的网格搜索参数。

集成层：Validation 反比 MAPE 加权，权重在 Test 评估前锁定，Test 不参与权重计算。

**结果（1 分钟）**

Final Test：XGBoost MAPE 0.36%、LSTM MAPE 0.44%、Ensemble MAPE 0.36%、R² 0.57。

Walk-forward（2021/2022/2023 各 12 个月）：XGBoost Mean MAPE 1.60%、LSTM Mean MAPE 1.41%、Naive 1.02%——复杂模型在多年稳健性测试中并未显著超过 Naive baseline，这反映了 132 点的样本约束。

**结尾（30 秒）**

项目展示了完整的数据获取 → 严格评估 → leakage 防控 → 集成学习工程实现 → 稳健性验证的端到端流程。所有数字都经过 9 项单元测试和 4 项 leakage 审计。技术栈：Python · akshare · Prophet · XGBoost · PyTorch · Streamlit。

---

## 15 分钟深度讲稿重点

详细见 `docs/interview_script_ml.md`。

要点：
- 为什么选择 PPI 作为预测目标
- 为什么 4 个统计基准必须存在（不是凑数）
- LSTM 在小样本下的局限性（132 点不足以让 LSTM 优于树模型）
- XGBoost 因果特征工程细节（shift(1) 修复 leakage）
- Ensemble 权重设计的逻辑（反比 MAPE）
- 为什么 Final Test 在低波动期结果不能过度解读
- Walk-forward 与 Final Test 的不同目的
- XGBoost subprocess isolation 工程细节

---

## 重要禁忌（写简历时不要做的事）

**禁止使用**：
- 集成 MAPE = 0.24%（旧 leaky 实验，已废弃）
- 集成比最强 XGBoost 低 15%（同上）
- LSTM R² = -0.74 → 0.61（旧 TF 时代）
- XGBoost Test MAPE = 0.283%（旧 leaky）
- 44 个手工估算数据点（已删除）
- 旧 2026 forecast 数字（98.9 / 106.5 / 110.4 / 116.0）
- "准确率 99.65%" 这类宣传性数字
- "预测准确" 而非 "MAPE"

**只使用**：
- 132 真实月度观测（akshare）
- 7 模型对比
- P0.5 Final Test 真实数字
- P0.6 Walk-forward 真实数字
- 完整 leakage 防控流程
- "PPI forecasting" 而非 "工程造价预测"
- MAPE / MAE / RMSE / R²（不使用"准确率"）
- "Walk-forward validation" 而非 "模型准确"
