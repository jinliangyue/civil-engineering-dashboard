"""
LSTM 超参网格搜索模块
功能：用 TimeSeriesSplit 时间序列交叉验证找最优 LSTM 超参
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

策略：
- 网格：units [16, 32, 64] × dropout [0.1, 0.2] × seq_length [6, 12, 18]
- 共 18 组合 × 3 折 CV = 54 次训练
- 评估指标：MAPE（越小越好）
- 时间序列 CV 避免未来信息泄漏
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import warnings
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_lstm_model(seq_length: int, units: int, dropout: float, lr: float = 0.001):
    """构建单个 LSTM 模型"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel('ERROR')

    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(units, return_sequences=(units >= 32)),
        Dropout(dropout),
        LSTM(units // 2 if units >= 32 else units, return_sequences=False),
        Dropout(dropout),
        Dense(8, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model


def train_one_combo(
    values: np.ndarray,
    seq_length: int,
    units: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    n_splits: int = 3,
) -> Dict:
    """
    训练单个超参组合 + 时间序列 CV 评估
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    if len(values) < seq_length + n_splits * 6 + 1:
        return {'status': 'error', 'reason': '数据不足', 'mape': np.inf}

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

        # 构造序列
        def make_seq(data):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length].reshape(-1, 1))
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        try:
            X_train, y_train = make_seq(train_scaled)
            X_test, y_test = make_seq(test_scaled)
            if len(X_train) < 4 or len(X_test) < 4:
                continue

            model = build_lstm_model(seq_length, units, dropout, lr)
            # Early stopping 减少训练时间
            from tensorflow.keras.callbacks import EarlyStopping
            es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = test_vals[seq_length:]
            # MAPE
            mask = y_true != 0
            if mask.sum() == 0:
                continue
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            mape_scores.append(mape)
            # 清理防止内存泄漏
            from tensorflow.keras import backend as K
            K.clear_session()
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
    LSTM 超参网格搜索

    默认网格：
    - units: [16, 32, 64]
    - dropout: [0.1, 0.2]
    - seq_length: [6, 12, 18]
    - lr: 0.001 固定
    = 18 组合 × 3 折 = 54 次训练
    """
    if param_grid is None:
        param_grid = {
            'units': [16, 32, 64],
            'dropout': [0.1, 0.2],
            'seq_length': [6, 12, 18],
            'lr': [0.001],
        }

    values = df[target_col].values.astype(float)

    # 生成所有组合
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    total = len(combos)

    logger.info(f'网格搜索: {total} 组合 × {n_splits} 折 = {total * n_splits} 次训练')

    results = []
    t_start = time.time()
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        result = train_one_combo(
            values,
            seq_length=params['seq_length'],
            units=params['units'],
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

    # 排序：成功的在前，按 MAPE 升序
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
        print('\n=== 网格搜索结果 Top 5 ===')
        for r in result['all_results'][:5]:
            print(f"MAPE={r['mape']:.4f}% | {r['params']}")
        print(f"\n最优: {result['best_params']}")
        print(f"用时: {result['total_time_seconds']:.0f}s")
