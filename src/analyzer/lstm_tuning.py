"""
LSTM 超参网格搜索模块（PyTorch 版）
功能：用 TimeSeriesSplit 时间序列交叉验证找最优 LSTM 超参
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

网格：hidden_size [32, 64, 128] × dropout [0.1, 0.2] × seq_length [6, 12] × num_layers [1, 2]
评估指标：MAPE（越小越好）
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


class _LSTMModel:
    """PyTorch LSTM 模型包装"""
    pass


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
    """训练单个超参组合 + 时间序列 CV 评估"""
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

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_vals.reshape(-1, 1)).flatten()

        try:
            # 构造序列
            def make_seq(data):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                    y.append(data[i + seq_length])
                return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

            X_train, y_train = make_seq(train_scaled)
            X_test, y_test = make_seq(test_scaled)
            if len(X_train) < 4 or len(X_test) < 4:
                continue

            X_train_t = torch.from_numpy(X_train).unsqueeze(-1)
            y_train_t = torch.from_numpy(y_train).unsqueeze(-1)
            X_test_t = torch.from_numpy(X_test).unsqueeze(-1)

            model, optimizer, criterion = _build_lstm_model(hidden_size, num_layers, dropout, lr)
            # 训练 + Early Stopping
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

            # 预测
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
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    param_grid: Dict = None,
    epochs: int = 50,
    batch_size: int = 8,
    n_splits: int = 3,
) -> Dict:
    """
    LSTM 超参网格搜索（PyTorch 版）
    默认网格：hidden_size [32, 64, 128] × dropout [0.1, 0.2] × seq_length [6, 12] = 12 组合 × 3 折 = 36 训练
    """
    if param_grid is None:
        param_grid = {
            'hidden_size': [32, 64, 128],
            'dropout': [0.1, 0.2],
            'seq_length': [6, 12],
            'num_layers': [2],
            'lr': [0.001],
        }

    values = df[target_col].values.astype(float)

    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    total = len(combos)

    logger.info(f'网格搜索: {total} 组合 × {n_splits} 折 = {total * n_splits} 次训练')

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
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ppi_monthly import load_monthly_ppi

    df = load_monthly_ppi()
    if df.empty:
        print('没有月度数据')
    else:
        print(f'数据: {len(df)} 月')
        result = grid_search_lstm(df, epochs=50, n_splits=3)
        print('\n=== Top 5 超参组合 ===')
        for r in result['all_results'][:5]:
            print(f"MAPE={r['mape']:.4f}% | {r['params']}")
        print(f'\n最优: {result["best_params"]}')
        print(f'用时: {result["total_time_seconds"]:.0f}s')
