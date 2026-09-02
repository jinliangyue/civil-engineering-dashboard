"""
LSTM 超参网格搜索模块（Phase 2 v3.1 严格版）

功能：用 TimeSeriesSplit 时间序列交叉验证找最优 LSTM 超参

数据边界不变量（Phase 2 v3.1 冻结）：
- Grid Search 严格只用 Train 段（2015-01 ~ 2021-12 = 84 月）
- 严禁访问 Validation (2022-01 ~ 2023-12) 或 Final Test (2024-01 ~ 2025-12)
- Scaler 只在 train_df 上 fit（每个 fold 的 train_idx 上 fit）
- Sequence/window 生成只用过去值（不泄漏未来）

历史问题（修复前）：
- 旧版本接受任意长度 df
- 如果传入完整 132 点，TimeSeriesSplit n_splits=3 在 132 点上：
  - Fold 3 test = idx [99:132] = 2023-05 ~ 2025-12（含 Final Test 段）
  - best_params 实际上「看过」Test 数据
  - 旧报告的 R² = -0.74 → 0.61 因此不可信

修复（Phase 3 P0.3）：
- 函数入口强校验：传入 df 长度 > 84 时抛 ValueError
- 强制打印数据边界（start / end / N）让调用方看见
- 默认 n_splits=3 保留（每折在 84 点上划分，全部 < 2022-01）

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging
import warnings
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === 硬编码：Train 段最大允许长度（84 月 = 2015-01 ~ 2021-12）===
# 这是 Phase 2 v3.1 冻结的边界
TRAIN_MAX_LENGTH = 84


def _build_lstm_model(hidden_size: int, num_layers: int, dropout: float, lr: float):
    """构建 PyTorch LSTM 模型"""
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def _train_one_combo(
    values: np.ndarray,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    n_splits: int = 3,
) -> Dict:
    """
    训练单个超参组合 + 时间序列 CV 评估

    内部使用 TimeSeriesSplit 在传入的 values 上划分。
    **关键约束**：调用方必须确保 values 仅包含 Train 段数据。
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    if len(values) < seq_length + n_splits * 6 + 1:
        return {'status': 'error', 'reason': '数据不足', 'mape': np.inf}

    torch.manual_seed(42)
    np.random.seed(42)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mape_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(values)):
        if len(test_idx) < seq_length + 1:
            continue
        train_vals = values[train_idx]
        test_vals = values[test_idx]

        # === Scaler fit 只在 train 上 ===
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_vals.reshape(-1, 1)).flatten()

        try:
            # 构造序列（X 用过去 seq_length 个点 → 不泄漏未来）
            def make_seq(data):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length].reshape(-1, 1))
                    y.append(data[i + seq_length])
                return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

            X_train, y_train = make_seq(train_scaled)
            X_test, y_test = make_seq(test_scaled)
            if len(X_train) < 4 or len(X_test) < 4:
                continue

            X_train_t = torch.from_numpy(X_train)
            y_train_t = torch.from_numpy(y_train)
            X_test_t = torch.from_numpy(X_test)

            model, optimizer, criterion = _build_lstm_model(hidden_size, num_layers, dropout, lr)
            # 训练 + Early Stopping（用训练集 loss，不接触 test）
            best_loss = float('inf')
            best_state = None
            patience = 10
            no_improve = 0
            for epoch in range(epochs):
                model.train()
                perm = np.random.permutation(len(X_train_t))
                epoch_loss = 0
                for i in range(0, len(X_train_t), batch_size):
                    idx = perm[i:i + batch_size]
                    xb = X_train_t[idx]
                    yb = y_train_t[idx]
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(xb)
                epoch_loss /= len(X_train_t)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)

            # 测试集评估（每个 fold 的 test_idx 来自 values 内部）
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_test_t).numpy().flatten()
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = test_vals[seq_length:]
            mask = y_true != 0
            if mask.sum() == 0:
                continue
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            mape_scores.append(mape)
        except Exception as e:
            logger.debug(f'fold {fold_idx} 失败: {e}')
            continue

    if not mape_scores:
        return {'status': 'error', 'reason': '所有 fold 都失败', 'mape': np.inf}

    return {
        'status': 'success',
        'mape': float(np.mean(mape_scores)),
        'mape_std': float(np.std(mape_scores)),
        'mape_per_fold': mape_scores,
        'n_folds': len(mape_scores),
    }


def grid_search_lstm(
    train_df: pd.DataFrame,
    target_col: str = 'ppi_index',
    param_grid: Dict = None,
    epochs: int = 50,
    batch_size: int = 8,
    n_splits: int = 3,
    strict_train_only: bool = True,
    max_train_length: int = TRAIN_MAX_LENGTH,
) -> Dict:
    """
    LSTM 超参网格搜索（Phase 2 v3.1 严格版）

    数据边界（必须遵守）：
    - train_df 必须只包含 Train 段（2015-01 ~ 2021-12 = 84 月）
    - 严禁传入 Validation 或 Final Test 段数据
    - 超参选择基于 train_df 内部的 TimeSeriesSplit CV

    Args:
        train_df: 仅包含 Train 段数据的 DataFrame
        target_col: 目标列名（默认 'ppi_index'）
        param_grid: 超参网格（默认 12 组合 × 3 折 = 36 训练）
        epochs: 训练轮数
        batch_size: 批大小
        n_splits: TimeSeriesSplit 折数
        strict_train_only: 强校验（默认 True；传入 > max_train_length 时抛 ValueError）
        max_train_length: 强校验阈值（默认 84）

    Returns:
        best_params: 最优超参组合
        best_mape: CV 平均 MAPE

    Raises:
        ValueError: strict_train_only=True 且 train_df 长度 > max_train_length
    """
    if param_grid is None:
        param_grid = {
            'hidden_size': [32, 64, 128],
            'dropout': [0.1, 0.2],
            'seq_length': [6, 12],
            'num_layers': [2],
            'lr': [0.001],
        }

    # === 数据边界强校验 ===
    if strict_train_only and len(train_df) > max_train_length:
        raise ValueError(
            f"train_df 长度 = {len(train_df)} > {max_train_length}。\n"
            f"Phase 2 v3.1 实验设计：Grid Search 严格只用 Train (2015-2021, 84 月)。\n"
            f"如果传入完整 132 点，TimeSeriesSplit Fold 3 会含 Final Test 段 → 数据泄漏。\n"
            f"请先调用 train_df = df[df.date <= '2021-12-31'] 再传入。\n"
            f"如确需覆盖此校验（不推荐），请设 strict_train_only=False。"
        )

    # === 强制打印数据边界 ===
    train_start = train_df['date'].min().strftime('%Y-%m')
    train_end = train_df['date'].max().strftime('%Y-%m')
    n = len(train_df)
    logger.info(f"=== LSTM Grid Search 数据边界 ===")
    logger.info(f"  Train start: {train_start}")
    logger.info(f"  Train end:   {train_end}")
    logger.info(f"  N points:    {n}")
    logger.info(f"  Strict:      {strict_train_only}")
    if n > max_train_length:
        logger.warning(
            f"  ⚠️  N ({n}) > max_train_length ({max_train_length}) "
            f"——可能引入 Validation/Test 泄漏"
        )

    values = train_df[target_col].values.astype(float)

    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    total = len(combos)

    logger.info(f"网格搜索: {total} 组合 × {n_splits} 折 = {total * n_splits} 次训练")

    results = []
    t_start = time.time()
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        result = _train_one_combo(
            values,
            seq_length=params['seq_length'],
            hidden_size=params['hidden_size'],
            num_layers=params.get('num_layers', 2),
            dropout=params['dropout'],
            lr=params['lr'],
            epochs=epochs,
            batch_size=batch_size,
            n_splits=n_splits,
        )
        result['params'] = params
        results.append(result)
        elapsed = time.time() - t_start
        eta = elapsed / i * (total - i)
        status = result['status']
        mape_str = f"{result.get('mape', np.inf):.4f}" if status == 'success' else 'FAIL'
        logger.info(f'[{i}/{total}] {params} → MAPE={mape_str}% ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)')

    results.sort(key=lambda x: (x['status'] != 'success', x.get('mape', np.inf)))

    best = results[0] if results and results[0]['status'] == 'success' else None
    return {
        'status': 'success' if best else 'error',
        'best_params': best['params'] if best else None,
        'best_mape': best['mape'] if best else np.inf,
        'all_results': [
            {
                'params': r['params'],
                'mape': r.get('mape', np.inf),
                'mape_std': r.get('mape_std', 0),
                'n_folds': r.get('n_folds', 0),
                'status': r['status'],
            }
            for r in results
        ],
        'total_time_seconds': time.time() - t_start,
        'n_combinations': total,
        'train_boundary': {
            'start': train_start,
            'end': train_end,
            'n_points': n,
        },
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ppi_monthly import load_monthly_ppi

    df_full = load_monthly_ppi()
    if df_full.empty:
        print('没有月度数据')
    else:
        # 严格切片到 Train 段
        df_train = df_full[df_full['date'] <= '2021-12-31'].copy()
        print(f'完整数据: {len(df_full)} 月')
        print(f'Train 切片: {len(df_train)} 月 ({df_train["date"].min()} ~ {df_train["date"].max()})')
        print()
        print('=== P0.3 测试 1: 严格模式 (strict_train_only=True) ===')
        try:
            result = grid_search_lstm(df_train, epochs=50, n_splits=3)
            print('\nTop 3 超参组合：')
            for r in result['all_results'][:3]:
                print(f"  MAPE={r['mape']:.4f}% | {r['params']}")
            print(f"\n最优: {result['best_params']} (MAPE={result['best_mape']:.4f}%)")
        except ValueError as e:
            print(f'ERROR: {e}')

        print()
        print('=== P0.3 测试 2: 边界校验（传入 132 点应报错）===')
        try:
            result = grid_search_lstm(df_full, epochs=50, n_splits=3)
            print('ERROR: 应该报错但没报！')
        except ValueError as e:
            print(f'✓ 正确拒绝: {str(e)[:120]}')
