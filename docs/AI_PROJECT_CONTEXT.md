# AI Project Context Snapshot

> **目的**：本文件供新 Claude Code 会话在没有聊天历史的情况下快速理解项目状态。
> **生成时间**：2026-09-03
> **当前 Git HEAD**：`251decf`（P0.9.5 环境锁定 commit · 历史章节中的旧 HEAD 引用以本行为准）
> **最近追加**：§15 P0.9.4（Cloud runtime UNKNOWN + 架构决策）· §16 P0.9.5（环境锁定，2026-09-03 commit `251decf`）

---

## 1. 项目身份

- **名称**：中国工业 PPI 月度时间序列分析与预测平台
- **仓库**：`jinliangyue/civil-engineering-dashboard`
- **在线 Demo**：https://civil-engineering-ppi.streamlit.app/
- **本地目录**：`~/Desktop/Claude code/civil-engineering-dashboard/`
- **作者**：jinliangyue（笔名"十八"，民企二本土木准大四）
- **用途**：2026 秋招简历项目

---

## 2. 正式数据

- **唯一数据源**：`data/raw/工业PPI_全国月度_2015-2025.csv`
- **数据来源**：akshare `macro_china_ppi()` 间接从国家统计局月度发布抓取
- **时间范围**：2015-01 ~ 2025-12
- **数据点数**：**132**（真实月度观测）

**已删除**：
- 4 行业 × 11 年 = 44 个手工估算年度数据（P0.1 commit `587f9c6` 删除）
- `generate_fallback.py` 中的 `FALLBACK_PPI` 字典

---

## 3. P0.5 实验设计（已冻结 LOCKED BASELINE）

### 3.1 数据划分

```
Train       2015-01 ~ 2021-12  84 points
Validation  2022-01 ~ 2023-12  24 points
Final Test  2024-01 ~ 2025-12  24 points
Total                        132 points
```

### 3.2 流程

```
Train (84) → Validation (24) → 锁定 Ensemble weights
       → Train + Validation (108) → Final Test (24)
```

### 3.3 P0.5 Final Test 正式结果（2024-01 ~ 2025-12, 24 月 OOS）

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Naive | 0.3583 | 0.4805 | 0.3667% | 0.5248 |
| Seasonal Naive | 1.2625 | 1.7599 | 1.2904% | -5.3757 |
| MA | 0.5903 | 0.7432 | 0.6040% | -0.1371 |
| SES | 0.5507 | 0.6948 | 0.5633% | 0.0062 |
| Prophet | 11.7556 | 12.4687 | 12.0476% | -319.0508 |
| XGBoost | 0.3482 | 0.4824 | **0.3558%** | 0.5209 |
| LSTM | 0.4288 | 0.5224 | **0.4387%** | 0.4381 |
| **Ensemble** | **0.3473** | **0.4589** | **0.3551%** | **0.5664** |

### 3.4 Ensemble 权重（来自 Validation，Test 前锁定）

```
Naive           0.28189
Seasonal Naive  0.03278
MA              0.15227
SES             0.11433
Prophet         0.00947
XGBoost         0.24592
LSTM            0.16333
```

公式：`raw_weight_i = 1 / Validation_MAPE_i; weight_i = raw_weight_i / Σ raw_weight`（无 round，保留完整 float 精度）

---

## 4. P0.6 Walk-forward 实验（已冻结 LOCKED BASELINE）

### 4.1 三个 Expanding Folds

| Fold | Train | Test |
|---|---|---|
| F1 | 2015-01 ~ 2020-12 (72) | 2021-01 ~ 2021-12 (12) |
| F2 | 2015-01 ~ 2021-12 (84) | 2022-01 ~ 2022-12 (12) |
| F3 | 2015-01 ~ 2022-12 (96) | 2023-01 ~ 2023-12 (12) |

### 4.2 P0.6 Mean MAPE ± Std

| Model | Mean MAPE | Std |
|---|---:|---:|
| Naive | 1.0192% | 0.2561% |
| Seasonal Naive | 7.9091% | 0.8075% |
| MA | 1.8170% | 0.5036% |
| SES | 2.4818% | 0.7270% |
| Prophet | 12.1357% | 2.1644% |
| XGBoost | 1.5958% | 0.8992% |
| LSTM | 1.4087% | 0.5024% |

**P0.6 不重新计算 Ensemble 权重**——沿用 P0.5 锁定的 Validation weights。

---

## 5. 当前模型列表（7 个）

| # | Model | Type |
|:--:|---|---|
| 1 | Naive | Baseline (y[t+1]=y[t]) |
| 2 | Seasonal Naive | Baseline (y[t+1]=y[t-12]) |
| 3 | Moving Average | Baseline (window=3) |
| 4 | SES | Baseline (alpha=0.3) |
| 5 | Prophet | Additive trend + yearly seasonality |
| 6 | XGBoost | 15 causal features + subprocess isolation |
| 7 | LSTM (PyTorch) | 2-layer + rolling one-step-ahead |

---

## 6. 数据泄漏防护原则（必须严格遵守）

1. **Hyperparameter tuning 只能使用 Train**（LSTM grid search 严格只接收 ≤84 点）
2. **Ensemble 权重只能来自 Validation**（Test MAPE 严禁参与权重计算）
3. **Final Test 不得参与任何调参 / 权重计算 / 模型选择**
4. **Final Test 只能用于最终一次评估**
5. **2026 Future Forecast 与 Final Test 独立**（Future Forecast 不能复用 Final Test 训练数据）
6. **不得重新引入 fallback 数据**（44 点手工估算已永久删除）
7. **不得重新引入旧的 0.24% 等数字**（属于 Deprecated/Historical Results）

---

## 7. 已完成阶段

| 阶段 | Commit | 内容 |
|---|---|---|
| Phase 2 v3.1 frozen | `ba60aae` | 实验设计冻结 + checkpoint tag |
| P0.1 | `587f9c6` | 删除 fallback 数据 |
| P0.2 | `79e9121` | 统一评估指标模块 |
| P0.3 | `42ae111` | LSTM 调参严格隔离 Train |
| P0.4 | `c087faa` + `30788cc` | LSTM rolling 24 步完整 |
| P0.5 | `6a901b2` | Validation-weighted Ensemble |
| P0.6 | `9a64d22` | Walk-forward Validation |
| P0.7 audit | (无 commit) | 结果审计 + 可信度分级 |
| P0.8 docs | `8583653` | README + docs 重构 |
| P0.8.1 docs | `1de7919` | 剩余文档一致性清理 |
| P0.9 audit | (无 commit) | Reproducibility 审计 |
| P0.9.1 cleanup | `1fcddf8` | 工程清理（fallback/绝对路径/ml_models 删除） |

---

## 8. 当前已知限制

1. **样本量**：仅 132 月度观测；Walk-forward 每 Fold 仅 12 个 Test 点
2. **单变量时间序列**：未加入宏观外生变量（PMI / CPI / 能源 / 汇率）
3. **2024-2025 低波动区间**：Final Test 处于 PPI 低波动期（range 1.7-2.1），低 MAPE 应结合 Naive baseline 解读
4. **P0.6 Walk-forward 中复杂模型未稳定击败 Naive**：所有模型均值在 0.5% MAPE 范围内
5. **无预测区间 / 不确定性估计**
6. **本机 Python 3.9 环境缺 streamlit/plotly**（不影响 requirements.txt 完整性，runtime.txt 指定 3.12）
7. **Python 3.12.0b3 + pip 26 有 packaging bug**（影响 P0.9.2 干净环境验证——已报告但未修复）

---

## 9. 明确禁止修改的实验结果（LOCKED BASELINE）

以下数字**绝对不得修改、不得更新、不得"调到一样"**：

```
P0.5 Ensemble Test MAPE  = 0.3551%
P0.5 Ensemble Test R²    = 0.5664

P0.5 XGBoost Test MAPE   = 0.3558%
P0.5 LSTM Test MAPE      = 0.4387%

P0.6 Naive Mean MAPE      = 1.0192%
P0.6 XGBoost Mean MAPE   = 1.5958%
P0.6 LSTM Mean MAPE      = 1.4087%
```

**已废弃 / 不得再作为正式结果宣传的数字**：
- 0.24% / 0.241%（集成 MAPE，经验估算 best × 0.85）
- 0.283% / 0.7282（XGBoost 旧版 rolling leakage）
- -0.74 → 0.61（旧 TF 时代 LSTM grid search 含 Final Test）
- 15%（集成提升，旧 leaky）
- 44 个手工估算数据点（已删除）
- 2026 forecast 98.9 / 106.5 / 110.4 / 116.0（来自已删除的 fallback）

---

## 10. P0.9.2 状态：尚未完成

**P0.9.2（Python 3.12 Clean-Environment Verification）尚未完成**。

已尝试路径：
- Python 3.12.0b3（系统自带 beta）创建 venv ✓
- `pip install -r requirements.txt` ✗ **失败**：pip 26.2.1 在 Python 3.12.0b3 上有已知 packaging bug（解析 setuptools "0.dev0" 失败）

未尝试路径：
- 用 Homebrew 装 Python 3.13 stable
- 用 ensurepip 装 pip < 24（同样触发 bug）
- 手动 patch venv 内 packaging/_parser.py（已部分成功 `_MIN_VERSION: 0.0.0`）

**P0.9.2 不是当前工作目录本机问题**——requirements.txt + runtime.txt（Python 3.12）是声明正确的，是 macOS Apple 自带的 3.12.0b3 beta 与 pip 26 的环境兼容问题。

---

## 11. 关键目录结构

```
civil-engineering-dashboard/
├── README.md
├── requirements.txt
├── runtime.txt                       # python-3.12
├── .gitignore
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/工业PPI_全国月度_2015-2025.csv
├── docs/
│   ├── AI_PROJECT_CONTEXT.md        # (this file)
│   ├── PROJECT_STATUS.md
│   ├── resume_description.md
│   ├── LSTM_TUNING_RESULTS.md
│   ├── DEPLOYMENT.md
│   ├── DATA_INPUT_SPEC.md
│   ├── data_sources.md
│   ├── WORKFLOW.md
│   └── interview_script_ml.md
├── scripts/
│   ├── fetch_ppi.py
│   ├── generate_fallback.py          # DEPRECATED · 不要调用
│   └── run_pipeline.py
└── src/
    ├── ppi_monthly.py                # 月度数据 loader
    ├── data_loader.py                 # 旧年度 loader（少量引用）
    ├── data_cleaner.py               # 旧年度 cleaner（少量引用）
    └── analyzer/
        ├── monthly_lstm.py            # Prophet + XGBoost + LSTM（PyTorch）
        ├── lstm_tuning.py             # 网格搜索（Train only 强校验）
        └── ensemble.py                # Validation-weighted ensemble
    └── evaluation/
        ├── metrics.py                 # MAPE / MAE / RMSE / R²
        ├── test_metrics.py            # 9/9 单元测试
        └── walk_forward.py            # P0.6
```

---

## 12. 关键命令（下次复现用）

```bash
# Metrics tests (9/9 单元测试)
python3 -m src.evaluation.test_metrics

# P0.5 reproduction
python3 -m src.analyzer.ensemble

# P0.6 reproduction
python3 -m src.evaluation.walk_forward

# Load 132 monthly points
python3 -c "from src.ppi_monthly import load_monthly_ppi; print(load_monthly_ppi().shape)"

# Streamlit
streamlit run app/streamlit_app.py
```

---

## 13. 下一步唯一任务

**完成 P0.9.2（Python 3.12 干净环境复现验证）**。

具体步骤：
1. 解决 Python 3.12 + pip 26 的 packaging bug（可能需要：Python 3.13 stable / 手动 patch venv）
2. 在干净 Python 3.12 venv 装 requirements.txt
3. 跑 `python3 -m src.evaluation.test_metrics`（必须 9/9 PASS）
4. 跑 `python3 -m src.analyzer.ensemble`（必须复现 P0.5 数字，atol ≤ 1e-4）
5. 跑 `python3 -m src.evaluation.walk_forward`（必须复现 P0.6 数字，atol ≤ 1e-4）
6. 启动 Streamlit（确认无 ImportError / fallback 调用 / 硬编码路径）
7. 删除临时 venv
8. 确认 working tree clean

**不允许**：修改实验代码 / 修改 requirements.txt / 修改 P0.5 / P0.6 数字 / 重新训练。

如果数字不一致：记录 actual vs expected diff，不修改代码，报告 NEEDS REVIEW。

---

## 14. 当前 working tree 已知状态

- HEAD = `1fcddf8`
- 有未 tracked 的 `.venv-p093/`（Python 3.12 临时 venv，P0.9.2 验证用）
- `.venv-p093` 未被 `.gitignore` 显式 ignore（P0.9.1 未添加，venv 名带 `.` 前缀未被 `venv/` pattern 覆盖）

P0.9.2 完成后应删除 `.venv-p093/` 并确认 working tree clean。

---

**本文件由 Claude Code 在 P0.9.1 完成后、P0.9.2 验证前生成，作为新 Claude Code 会话的项目状态压缩快照。**（P0.9.4/P0.9.5 仅追加章节与修正过时 HEAD 标注，未改动正文内容。）

---

## 15. P0.9.4 审计（2026-09-03 · 只读 · 无文件变更）

### 15.1 结论

- **Q1 Cloud 实际 Python 版本：UNKNOWN / NOT OBSERVABLE**。live app 在公开面返回 303 登录墙（`/-/login`、`/_stcore/health` 均 303）；GitHub 无 Actions runs、无 deployments 记录（gh CLI 已登录 jinliangyue）；`runtime.txt = python-3.12` 只是 repository configuration，且仓库内两个 commit（`d5d7a48` 称 3.12 稳定、`3de3695` 称 runtime.txt 已废弃）叙述矛盾。任何「Cloud = 3.x」的写法均无证据。
- **Q2 推荐架构：B（Research / Deployment 分离）**。两个环境对象、两套锁文件、两套约束，对 UNKNOWN 稳健。
- **特别判断**：不把 research 强升 3.12。P0.5/P0.6 只在 3.9 全量复现；3.12 已有实测漂移（R3：Prophet 12.0476% → 11.5787%，Ensemble 0.3551% → 0.3540%）；reproducibility > demo 环境统一。迁移 3.9 baseline 需要独立的 lock 轮。

### 15.2 铁律（源自 P0.9.4/P0.9.5 规范）

> **Never use deployment environment changes to silently replace formal research results.**
> 正式实验结果的 reproducibility 高于线上 demo 环境统一。不得为「Cloud 更漂亮」牺牲 research reproducibility；不得为统一环境强升 3.9 baseline；不得为测试通过改模型；不得为报告好看隐藏 UNKNOWN。

---

## 16. P0.9.5 环境锁定（2026-09-03 · 已完成，见 docs/ENVIRONMENT.md）

### 16.1 交付文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `requirements-research.txt` | 新增 | 10 包全 pin（3.9 矩阵），P0.5/P0.6 唯一 reproduction baseline |
| `requirements-deploy.txt` | 新增 | documented compatibility baseline，顶部注明 Cloud runtime has not been independently verified |
| `scripts/environment_fingerprint.py` | 新增 | 版本 + Prophet 二进制 MD5；stdlib only；不输出敏感信息；缺包不 crash |
| `docs/ENVIRONMENT.md` | 新增 | 10 节：分层总览 / research 锁 / deploy baseline / Cloud UNKNOWN / 版本策略 / 指纹 / Prophet 证据 / Research vs Live Demo / 变更规则 / 其他事实 |

### 16.2 Research 环境（已正式锁定，9/3 复核）

Python 3.9.13（框架 build）+ pandas 2.3.3 / numpy 2.0.2 / scipy 1.13.1 / scikit-learn 1.6.1 / xgboost 2.1.4 / torch 2.8.0 / prophet 1.3.0 / cmdstanpy 1.3.0 / stanio 0.5.1 / holidays 0.83；streamlit/plotly 未装（验证过）。Prophet 指纹复核无 discrepancy：bundled CmdStan 2.37.0、`prophet_model.bin` MD5 72d9ae8b8f399727c6c5b2f7cfeb98e5、`prophet.stan` 971f6716…、`stanc` 632d992a…——与 P0.9.3 记录完全一致。现存唯一 research 安装 = 框架 Python 3.9 base（`.venv-p093-legacy` 副本已删）。

### 16.3 Formal results 状态（未改动，继续 LOCKED）

P0.5 Ensemble 0.3551% / 0.5664、XGBoost 0.3558%、LSTM 0.4387%；P0.6 Naive 1.0192% / XGBoost 1.5958% / LSTM 1.4087%（全表见 §3/§4）。未被 Cloud 数字覆盖，不因部署需求重训。

### 16.4 Deployment 状态

Cloud Python 仍为 UNKNOWN（§15.1 证据未变）。`requirements-deploy.txt` 是待验证 baseline，不声称已生效；Cloud 实际安装的仍是 unpinned `requirements.txt`。runtime.txt 保留 `python-3.12`（repository configuration only）。

### 16.5 Git 状态更正（相对本文档旧章节）

- 当前 HEAD = `251decf`（P0.9.5 环境锁 commit，2026-09-03）；local main 领先 `origin/main`（=`3de3695`）**14** 个 commit（13 个 research/修复 + 1 个 P0.9.5 环境锁）。
- 本地 research 修复（P0.5/P0.6/path fix）尚未在 origin/main 上 → Cloud 代码基线 = `3de3695`。**禁止自行 push**。
