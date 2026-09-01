# 工作流说明

> 你（用户）和 Claude Code 之间的协作流程。

---

## 总体流程

```
阶段 1（9/1-9/7）    数据调研 + 下载
阶段 2（9/8-9/14）   数据清洗 + 基础图表
阶段 3（9/15-9/21）  预测模型 + 模型评估
阶段 4（9/22-9/30）  仪表盘完善 + 部署 + 简历化
```

---

## 阶段 1：数据调研 + 下载（9/1-9/7）

### 你做什么

1. 按 `docs/data_sources.md` 调研 1-2 个数据源（建议先北京造价信息网）
2. 注册账号（如需要）
3. 下载 1 种材料 × 1 个地区 × 5 年的月度价格数据
4. 按 `docs/DATA_INPUT_SPEC.md` 规范整理成 CSV
5. 放进 `data/raw/` 目录

### Claude Code 做什么

1. 验证 CSV 字段完整性
2. 给你「数据可用性反馈」（哪些字段对、哪些缺）
3. 准备下一阶段的代码

### 交付物

- `data/raw/<材料>_<地区>_<时间>.csv`（你提供）
- Claude Code 给出「数据质量评估报告」

---

## 阶段 2：数据清洗 + 基础图表（9/8-9/14）

### 你做什么

1. 把 CSV 文件放进 `data/raw/`
2. 等 Claude Code 跑通清洗 + 出图

### Claude Code 做什么

1. 写 `src/data_loader.py` + `src/data_cleaner.py`
2. 跑数据清洗 → 输出到 `data/processed/`
3. 写 `src/analyzer/trend.py` + `src/visualizer/plotly_helpers.py`
4. 生成 5 种基础图表（趋势线 / 月度对比 / 同比环比 / 季节性热图 / 价格区间分布）
5. 写 README 初版 + 部署到 Streamlit Cloud 给你看 demo

### 交付物

- `data/processed/<材料>_<地区>_cleaned.csv`
- `data/processed/figures/` 下的 5-10 张图表
- Streamlit Cloud 在线 demo 链接
- GitHub 仓库初版

---

## 阶段 3：预测模型 + 模型评估（9/15-9/21）

### 你做什么

1. 看 demo + 给反馈（哪些图表不清楚 / 哪些分析想加）
2. 决定要不要加多材料对比

### Claude Code 做什么

1. 写 `src/analyzer/forecast.py`（Prophet 模型）
2. 模型评估（MAE / RMSE / MAPE / R²）
3. 多模型对比（Prophet vs ARIMA vs 简单线性回归）
4. 季节性分解（trend + seasonal + residual）
5. Streamlit 仪表盘加预测模块

### 交付物

- 预测结果 CSV + 图表
- 模型评估报告
- Streamlit 仪表盘加预测页面

---

## 阶段 4：仪表盘完善 + 部署 + 简历化（9/22-9/30）

### 你做什么

1. 测试 demo 完整体验
2. 提供简历项目栏初稿（如果有特定公司 JD 更好）
3. 提供联系方式（GitHub 仓库需要）

### Claude Code 做什么

1. 完善 Streamlit 多页仪表盘
2. 写 `docs/methodology.md`（方法论文档）
3. 写 `docs/resume_description.md`（简历描述模板）
4. 完整 README（项目背景 + 数据 + 方法 + 截图 + 部署 + 局限性）
5. 准备面试讲稿（5 分钟版 + 15 分钟版）
6. 部署到 Streamlit Cloud（公开访问）

### 交付物

- 完整 GitHub 仓库
- Streamlit Cloud 在线 demo
- 简历描述初稿
- 面试讲稿（5/15 分钟两版）

---

## 沟通节奏

**主动反馈**：你跑完任何一步后告诉我「我做了什么 + 有什么问题」
**我主动反馈**：每完成一阶段后给你交付清单 + 下一步建议
**阻塞处理**：如果你卡在某一步超过 2 小时，告诉我具体卡在哪，我们换方案

---

## 风险兜底

| 风险 | 兜底方案 |
|---|---|
| 数据源拿不到 5 种材料 | 立即切到 3 种材料 + 补备选数据源 |
| 预测模型 MAPE > 30% | 改用简单移动平均 + 注明「方法演示」 |
| Streamlit 部署失败 | 改用 GitHub Pages + 静态图表 |
| 时间不够 | 砍掉多材料对比，只保留单材料 + 单地区 |
| GitHub 仓库创建失败 | 用 Gitee 或本地 + 邮件交付 |

---

## Git 仓库（待办）

10 月 1 日前建好 GitHub 仓库（用户操作）：
1. 在 GitHub 创建新仓库 `civil-engineering-dashboard`
2. 本地 git init + git remote add origin
3. git push -u origin main
4. 部署 Streamlit Cloud 绑定 GitHub 仓库

Claude Code 会写好 .gitignore 和初始化脚本。

---

有问题随时问我。