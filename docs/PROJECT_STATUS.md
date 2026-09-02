# 项目完整状态报告（2026-09-03 P0.8 重构）

> 这是项目的完整快照，下次重启或换电脑也能从这里恢复所有上下文。
> **当前阶段**：P0.7 审计完成 + P0.8 文档清理完成

---

## 1. 项目身份

```
项目名：中国工业 PPI 月度时间序列分析与预测平台
仓库：jinliangyue/civil-engineering-dashboard
在线 Demo：https://civil-engineering-ppi.streamlit.app/
本地目录：~/Desktop/Claude code/civil-engineering-dashboard/
作者笔名：十八（民企二本土木准大四）
用途：2026 秋招简历项目
创建时间：2026-09-01
最新重构：2026-09-02 ~ 2026-09-03（P0.1 ~ P0.8）
```

---

## 2. 当前正式数据

```
数据文件：data/raw/工业PPI_全国月度_2015-2025.csv
数据源：akshare.macro_china_ppi() 间接从国家统计局月度发布抓取
时间范围：2015-01 ~ 2025-12（共 11 年）
数据点数：132 个真实月度观测
字段：date, ppi_index, yoy_pct, ytd_index
指数基准：上年同月 = 100
```

**已删除**：4 行业 × 11 年 = 44 个手工估算数据点（旧 fallback 已废弃）。

---

## 3. 正式实验划分（Phase 2 v3.1 冻结）

```
Train       2015-01 ~ 2021-12  84 points
Validation  2022-01 ~ 2023-12  24 points
Final Test  2024-01 ~ 2025-12  24 points
Total                        132 points
```

---

## 4. P0.5 正式实验结果（Final Test · 2024-01 ~ 2025-12）

| Model          |   MAE |   RMSE |    MAPE |        R² |
| -------------- | ----: | -----: | ------: | --------: |
| Naive          | 0.3583| 0.4805 | 0.3667% |    0.5248 |
| Seasonal Naive | 1.2625| 1.7599 | 1.2904% |   -5.3757 |
| MA             | 0.5903| 0.7432 | 0.6040% |   -0.1371 |
| SES            | 0.5507| 0.6948 | 0.5633% |    0.0062 |
| Prophet        |11.7556|12.4687 |12.0476% | -319.0508 |
| XGBoost        | 0.3482| 0.4824 | 0.3558% |    0.5209 |
| LSTM           | 0.4288| 0.5224 | 0.4387% |    0.4381 |
| **Ensemble**   |**0.3473**|**0.4589**|**0.3551%**|**0.5664** |

### Ensemble 权重（来自 Validation，Test 评估前锁定）

```
Naive           0.28189
Seasonal Naive  0.03278
MA              0.15227
SES             0.11433
Prophet         0.00947
XGBoost         0.24592
LSTM            0.16333
```

---

## 5. P0.6 Walk-forward 稳健性验证

### Folds

```
F1: Train 2015-2020 (72) → Test 2021 (12)
F2: Train 2015-2021 (84) → Test 2022 (12)
F3: Train 2015-2022 (96) → Test 2023 (12)
```

### Mean MAPE ± Std

| Model          |  Mean MAPE |   Std   |
| -------------- | ---------: | ------: |
| Naive          |   1.0192% | 0.2561% |
| Seasonal Naive |   7.9091% | 0.8075% |
| MA             |   1.8170% | 0.5036% |
| SES            |   2.4818% | 0.7270% |
| Prophet        |  12.1357% | 2.1644% |
| XGBoost        |   1.5958% | 0.8992% |
| LSTM           |   1.4087% | 0.5024% |

注：Walk-forward 未重新计算 Ensemble 权重（P0.5 权重来自固定 Validation）。

---

## 6. P0.5 vs P0.6 差异（重要解释）

**P0.5 测试期 2024-2025** 处于 PPI 低波动阶段（annual range 2.1 / 1.7）。
Naive baseline 在该期间 MAPE=0.37%，**接近 0.4% 已经非常接近理论极限**。

**P0.6 覆盖 2021 / 2022 / 2023** 包含 PPI 高位、回落、平稳三个阶段（2021 range 13.2）。

P0.5 的 Ensemble MAPE=0.3551% **不应被解读为复杂模型的预测能力**——它主要反映 2024-2025 区间本身的低波动性。

P0.6 的 Mean MAPE 更能反映模型在不同历史窗口下的稳健性。

**两套实验目的不同，不能简单比较数字大小。**

---

## 7. 已验证的工程实现

### XGBoost Subprocess 隔离

XGBoost 训练 + 预测在独立 Python 子进程（spawn 启动 / 60 秒 hard timeout / terminate 处理）执行。
主进程在子进程退出后才跑 LSTM，避免 XGBoost + PyTorch runtime 冲突。

### 严格 Causal Feature

XGBoost 所有 lag / rolling / yoy / mom 特征在 `build_features_monthly_causal` 中通过 `shift(1)` 保证只用过去值。

### Rolling One-step-ahead Prediction

XGBoost 和 LSTM 都用 one-step-ahead rolling prediction：
```
预测 t+1 → 用 history（含 train + 之前 test actual）
预测完成后 → 把 t+1 actual 加入 history
预测 t+2 → 用更新后的 history
```

### LSTM Hyperparameter Isolation

P0.3 已锁定 LSTM 调参严格只用 Train 84 点（P0.3 commit `42ae111`），不接触 Validation 或 Test。

---

## 8. Git History（主线）

```
9a64d22  feat: add walk-forward validation               P0.6
6a901b2  feat: build validation-weighted ensemble        P0.5
30788cc  fix: complete LSTM test rolling predictions     P0.4 修复
c087faa  refactor: make monthly models out-of-sample     P0.4
42ae111  refactor: isolate LSTM tuning to train data     P0.3
79e9121  feat: add unified evaluation metrics            P0.2
587f9c6  refactor: remove legacy PPI fallback data       P0.1
ba60aae  checkpoint: Phase 2 v3.1 frozen
```

---

## 9. 测试与质量保证

- `src/evaluation/metrics.py` 9/9 单元测试套件通过
- P0.5 / P0.6 leakage checks 全部 PASS（boundary / date alignment / XGB causal / LSTM causal / no Test contamination）
- 132 月度数据全部可追溯到 akshare → 国家统计局公开 PPI

---

## 10. 已废弃的旧结果（不得再用于正式展示）

以下数字来自旧版本（fallback 数据 / 旧 leaky 实现 / TF 时代 LSTM 调优），已不再适用：

- 集成 MAPE = 0.241%
- 集成比单一最强 XGBoost 低 15%
- LSTM R² = -0.74 → 0.61
- XGBoost Test MAPE = 0.283%
- XGBoost R² = 0.7282
- 4 行业 × 11 年 = 44 手工估算点
- 旧 2026 forecast: 98.9 / 106.5 / 110.4 / 116.0

这些数字已替换为本文第 4-5 节中的 P0.5/P0.6 真实数字。

---

## 11. 秋招投递清单（保留）

```
第一梯队（最高匹配）：中建 / 中铁 / 中交 / 中冶 / 中电建 / 能建
第二梯队：BIM 咨询 / 设计院 / 智慧工地公司
第三梯队：地方国企 / 大型房企工程岗
```

---

## 12. Limitations

1. 132 月度观测点限制
2. 单变量时间序列（未加入宏观外生变量）
3. 2024-2025 Test 处于 PPI 低波动阶段
4. Walk-forward 中复杂模型并未稳定击败 Naive baseline
5. 无预测区间 / 不确定性估计
6. 当前为方法论验证，不应解读为对未来 PPI 的保证

详见 README.md §Limitations。

---

## 13. Future Work

- 加入外生宏观变量（PMI / CPI / 能源价格 / 汇率 / 商品价格）
- 多变量时序模型
- 更长时间跨度的数据
- 更严格的 rolling-origin evaluation
- 预测区间 / 不确定性估计
- 2026 future-forecast monitoring（仅在验证 pipeline 完全可复现后）

---

## 14. 联系信息

```
作者笔名：十八
求职目标：中建 / 中铁 / 中交 系统 · BIM / 智慧工地 / 工程信息化岗位
GitHub：jinliangyue
Streamlit Demo：civil-engineering-ppi.streamlit.app
```

---

**本文件由 Claude Code 在 2026-09-03 重构，作为 P0.1~P0.8 完成后的项目状态备份。**
