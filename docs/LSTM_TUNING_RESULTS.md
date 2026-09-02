# Historical LSTM Experiments and Superseded Results

> **状态**：**已废弃 / 不再作为正式成果**
> **重写日期**：2026-09-03（P0.8 文档清理）
> **原因**：早期实验存在数据泄漏与已删除的 fallback 数据，结果不可信

---

## 1. 为什么这份文档被废弃

本文档记录的 LSTM 调优与集成学习结果来自以下已废弃的实现：

| 旧实现 | 问题 | 当前替代 |
|---|---|---|
| TensorFlow/Keras LSTM | Python 3.14 兼容性问题 | **PyTorch LSTM**（`src/analyzer/monthly_lstm.py`） |
| 在完整 132 点 df 上做 grid search | **grid search Fold 3 含 2024-2025 Final Test 数据** → 严重数据泄漏 | **Grid Search 严格只用 Train 84 月**（P0.3 commit `42ae111`） |
| in-sample 评估（fit 132 → predict 同一份） | MAPE 严重虚低 | **out-of-sample 评估**（fit train+val → predict test） |
| `ensemble_mape = best_mape × 0.85` 经验估算 | 不是真实计算 | **真实加权平均**：ensemble_pred = Σ wᵢ × preds[i]，再算 MAPE |
| 集成 MAPE = 0.241% | 来源于上述有缺陷的实现 | 详见下文 §3 |

**结论**：本文档保留为历史参考，但所有数字不再作为正式项目成果。

---

## 2. 旧实验曾报告的数字（已废弃）

| 旧数字 | 来源 | 实际原因 |
|---|---:|---|
| 集成 MAPE = 0.241% | 旧 ensemble.py `ensemble_mape = best_mape × 0.85` | **不是真实计算**，是经验估算 |
| 集成 R² = 0.7896 | 同上 | 同上 |
| 集成比单一最强 XGBoost 低 15% | `best × 0.85` | 同上 |
| LSTM R² = -0.74（默认超参） | 旧 TensorFlow，grid search 用完整 132 点 | grid search 看过了 Test 段，best_params 过拟合 |
| LSTM R² = 0.51（调优后） | 同上 | 同上 |
| XGBoost MAPE = 0.283% | 旧 rolling one-step-ahead，无 shift(1) 因果修复 | rolling 特征含当前 target → leakage |
| XGBoost R² = 0.7282 | 同上 | 同上 |
| 4 行业 × 11 年 = 44 个手工估算点 | `FALLBACK_PPI` 字典 | **已删除**（P0.1 commit `587f9c6`） |
| 2026 forecast: 98.9 / 106.5 / 110.4 / 116.0 | 旧 fallback 数据 + 旧线性回归 | 基础数据已删除，结果无意义 |

---

## 3. 当前正式 LSTM 结果（P0.5 + P0.6）

### 3.1 P0.5 Final Test（2024-01 ~ 2025-12，24 月 OOS）

**使用 P0.3 锁定参数**：`hidden_size=32, dropout=0.1, seq_length=6, num_layers=2, lr=0.001`

| 指标 | 值 |
|---|---:|
| MAE | 0.4288 |
| RMSE | 0.5224 |
| **MAPE** | **0.4387%** |
| R² | 0.4381 |

### 3.2 P0.6 Walk-forward Mean MAPE（3 expanding folds）

| Fold | MAPE |
|---|---:|
| F1 (2021) | 1.5558% |
| F2 (2022) | 1.9372% |
| F3 (2023) | 0.7332% |
| **Mean** | **1.4087%** |
| **Std** | 0.5024% |

### 3.3 P0.5 与 P0.6 LSTM 数字的合理解释

- **LSTM 在 P0.5（2024-2025）MAPE=0.44%**，与 XGBoost（0.36%）接近但稍逊。这反映了 **132 点的样本约束**——LSTM 的优势需要更大数据量。
- **LSTM 在 P0.6 Walk-forward Mean MAPE=1.41%**，与 XGBoost（1.60%）相当甚至略好——多年区间下 LSTM 的非线性建模能力部分发挥。
- **总体结论**：在小样本时序数据上，LSTM 与 XGBoost 表现接近。声称"LSTM 远优于树模型"在 132 点数据上不成立，**诚实解读实验结果**。
- 在 `docs/resume_description.md` 面试讲稿中明确说明："LSTM 在小样本下能力受限，Prophet 在月度季节性上表现更稳"。

---

## 4. P0.3 重写后的 LSTM 调优流程（当前正式版本）

### 4.1 网格空间（P0.3 commit `42ae111`）

```
网格：
- hidden_size: [32, 64, 128]
- dropout: [0.1, 0.2]
- seq_length: [6, 12]
- num_layers: [2]
- lr: [0.001]
= 12 组合 × 3 折时间序列 CV = 36 次训练
```

**严格边界**：
- Grid Search 只用 **Train 84 月**（2015-01 ~ 2021-12）
- TimeSeriesSplit 在 Train 内部划分（n_splits=3 → 每折 train ~21-63 月，test ~21 月）
- **Validation / Final Test 永不进入 Grid Search**
- `grid_search_lstm` 入口强校验：`len(df) > 84` 时抛 `ValueError`

### 4.2 当前锁定参数（P0.3 调优结果）

```python
LSTM_BEST_PARAMS = {
    'hidden_size': 32,
    'dropout': 0.1,
    'seq_length': 6,
    'num_layers': 2,
    'lr': 0.001,
}
```

**注意**：当前 PyTorch 默认参数实际就是 P0.3 调优结果，因此**无需重新 grid search**。

### 4.3 当前 Grid Search 输出（在 84 月 Train 上）

CV 3 折平均 MAPE = **5.16%**

> 说明：这是 **Train 内部 CV MAPE**，**不是 Final Test MAPE**。两者不可比。
> Final Test LSTM MAPE = 0.4387%（见 §3.1）。

---

## 5. 旧集成学习的真实计算（P0.5 替代）

### 5.1 旧集成（已废弃）

```
ensemble_mape_estimated = best_single_mape × 0.85
→ 0.283 × 0.85 = 0.241%
```

**问题**：不是真实计算，仅是经验估算。

### 5.2 新集成（当前正式 P0.5）

```python
# Validation 阶段
weights = inverse_mape_weights(val_mapes)
# 不做 round，保留完整 float 精度
# assert abs(sum(weights.values()) - 1.0) < 1e-8

# Test 阶段
ensemble_preds = sum(weight[name] * test_preds[name] for name in models)
ensemble_mape = mape(test_actuals, ensemble_preds)  # 真实计算
```

**真实 P0.5 Ensemble Test MAPE = 0.3551%**（基于 Validation 阶段锁定的 weights）

---

## 6. 总结

| 阶段 | 旧（已废弃） | 新（P0.5 / P0.6） |
|---|---|---|
| LSTM 框架 | TensorFlow/Keras | PyTorch |
| Grid Search 数据 | 完整 132 点（**泄漏**） | Train 84 点（P0.3 强校验） |
| 集成权重来源 | `best × 0.85` 估算 | Validation 真实反比 MAPE（test 不参与） |
| 集成 Test MAPE | 0.241%（假） | **0.3551%（真）** |
| LSTM Test R² | -0.74 → 0.51（泄漏） | **R² = 0.4381**（OOS 真） |
| XGBoost Test MAPE | 0.283%（rolling 泄漏） | **0.3558%**（causal 真） |

**新数字总体上升但全部真实可信**——这是 P0.1~P0.6 重构后的科学严谨性代价。

---

## 7. 参考

- P0.5 完整结果：`docs/PROJECT_STATUS.md` 第 4 节
- P0.6 完整结果：`docs/PROJECT_STATUS.md` 第 5 节
- 旧结果被替换的位置：`README.md`、`docs/PROJECT_STATUS.md`、`docs/resume_description.md`

---

**本文件由 Claude Code 在 2026-09-03 重构，明确标记所有旧 LSTM 调优与集成学习数字为已废弃历史结果。**
