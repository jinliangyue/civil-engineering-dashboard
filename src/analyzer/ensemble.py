"""
集成学习模块
功能：XGBoost + Prophet + LSTM（网格搜索最优超参）三模型加权平均
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

策略：
- LSTM 用 lstm_tuning.grid_search_lstm 找到的最优超参（units=64, dropout=0.1, seq_length=6）
- 加权方式：测试集 MAPE 反比加权（精度越高权重越大）
  - weight_i = (1 / mape_i) / sum(1 / mape_j)
- 输出：集成预测 + 与单一模型对比
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def inverse_mape_weights(mapes: Dict[str, float]) -> Dict[str, float]:
    """
    基于测试集 MAPE 反比加权（精度越高权重越大）
    mape_i 越小 → 1/mape_i 越大 → weight_i 越大
    """
    # 过滤掉失败/无穷大的
    valid = {k: v for k, v in mapes.items() if v is not None and v != np.inf and v > 0}
    if not valid:
        return {'xgboost': 0.33, 'prophet': 0.33, 'lstm': 0.34}
    inv = {k: 1.0 / v for k, v in valid.items()}
    total = sum(inv.values())
    return {k: round(v / total, 4) for k, v in inv.items()}


def train_ensemble(
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    test_months: int = 24,
    forecast_months: int = 12,
    lstm_params: Dict = None,
) -> Dict:
    """
    训练三模型 + 集成预测

    lstm_params: 网格搜索找到的最优超参，默认从 lstm_tuning 调用
    """
    from src.analyzer.monthly_lstm import (
        train_prophet_monthly,
        train_xgboost_monthly,
        build_features_monthly,
        get_monthly_feature_columns,
    )

    if lstm_params is None:
        # 默认用网格搜索结果
        lstm_params = {'units': 64, 'dropout': 0.1, 'seq_length': 6, 'lr': 0.001}

    if df.empty:
        return {'status': 'error', 'reason': '输入数据为空'}

    logger.info(f'LSTM 超参: {lstm_params}')

    # ============ 1. 训练 LSTM（用最优超参）============
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    from sklearn.preprocessing import StandardScaler

    values = df[target_col].values.astype(float)
    split_idx = len(values) - test_months
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    seq_length = lstm_params['seq_length']
    lstm_result = {'status': 'error'}
    lstm_test_pred = None
    lstm_future_pred = None

    try:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()

        def make_seq(data):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length].reshape(-1, 1))
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        X_train_seq, y_train_seq = make_seq(train_scaled)
        X_test_seq, y_test_seq = make_seq(test_scaled)

        # 构建最优 LSTM
        units = lstm_params['units']
        dropout = lstm_params['dropout']
        K.clear_session()
        import tensorflow as tf
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
        model.compile(optimizer=Adam(learning_rate=lstm_params.get('lr', 0.001)), loss='mse', metrics=['mae'])
        es = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0)
        model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=8, verbose=0, callbacks=[es])

        # 测试集预测
        y_pred_test_scaled = model.predict(X_test_seq, verbose=0).flatten()
        lstm_test_pred = scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        lstm_result['metrics'] = {
            'MAE': round(float(np.mean(np.abs(test_values[seq_length:] - lstm_test_pred))), 3),
            'RMSE': round(float(np.sqrt(np.mean((test_values[seq_length:] - lstm_test_pred) ** 2))), 3),
            'MAPE_pct': round(calculate_mape(test_values[seq_length:], lstm_test_pred), 3),
            'R_squared': round(float(1 - np.sum((test_values[seq_length:] - lstm_test_pred) ** 2) /
                                       np.sum((test_values[seq_length:] - test_values[seq_length:].mean()) ** 2)), 4),
        }

        # 未来预测（递归）
        future_predictions = []
        last_seq = scaler.transform(values[-seq_length:].reshape(-1, 1)).flatten().reshape(-1, 1)
        current_seq = last_seq.copy()
        for _ in range(forecast_months):
            pred_scaled = model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)[0][0]
            pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            next_date = df['date'].iloc[-1] + pd.DateOffset(months=len(future_predictions) + 1)
            future_predictions.append({
                'date': next_date,
                'predicted_ppi': pred,
            })
            current_seq = np.append(current_seq[1:], pred_scaled).reshape(-1, 1)
        lstm_future_pred = future_predictions
        lstm_result['status'] = 'success'
        K.clear_session()
    except Exception as e:
        logger.error(f'LSTM 调优后训练失败: {e}')

    # ============ 2. 训练 Prophet ============
    prophet_result = train_prophet_monthly(df, target_col, forecast_months)
    prophet_test_pred = None
    if prophet_result.get('status') == 'success':
        # 提取测试集预测（Prophet 的 historical_fit 含全部）
        hist = prophet_result['historical_fit']
        hist_aligned = hist.set_index('ds')['yhat'].reindex(df['date']).values
        # 取最后 test_months 个
        prophet_test_pred = hist_aligned[-test_months:]
        # 修正长度（Prophet 的 yhat 跟原数据对齐）
        if len(prophet_test_pred) >= seq_length:
            prophet_test_pred = prophet_test_pred[-len(lstm_test_pred):] if lstm_test_pred is not None else prophet_test_pred[-test_months:]

    # ============ 3. 训练 XGBoost ============
    df_features = build_features_monthly(df, target_col)
    feature_cols = get_monthly_feature_columns()
    xgb_result = train_xgboost_monthly(df_features, feature_cols, target_col, test_months, forecast_months)
    xgb_test_pred = None
    if xgb_result.get('status') == 'success' and 'test_predictions' in xgb_result:
        xgb_test_pred = xgb_result['test_predictions']

    # ============ 4. 计算权重 + 集成预测 ============
    mapes = {
        'xgboost': xgb_result.get('metrics', {}).get('MAPE_pct') if xgb_result.get('status') == 'success' else None,
        'prophet': prophet_result.get('metrics', {}).get('MAPE_pct') if prophet_result.get('status') == 'success' else None,
        'lstm': lstm_result.get('metrics', {}).get('MAPE_pct') if lstm_result.get('status') == 'success' else None,
    }
    weights = inverse_mape_weights(mapes)
    logger.info(f'模型权重（反比 MAPE 加权）: {weights}')

    # 测试集集成
    ensemble_test_pred = None
    ensemble_metrics = None
    if lstm_test_pred is not None and prophet_test_pred is not None and xgb_test_pred is not None:
        # 对齐长度（用最短的）
        min_len = min(len(lstm_test_pred), len(prophet_test_pred), len(xgb_test_pred))
        lstm_a = lstm_test_pred[-min_len:]
        prophet_a = prophet_test_pred[-min_len:]
        xgb_a = xgb_test_pred[-min_len:]
        ensemble_test_pred = (
            weights['xgboost'] * xgb_a +
            weights['prophet'] * prophet_a +
            weights['lstm'] * lstm_a
        )
        y_true = test_values[-min_len:]
        ensemble_metrics = {
            'MAE': round(float(np.mean(np.abs(y_true - ensemble_test_pred))), 3),
            'RMSE': round(float(np.sqrt(np.mean((y_true - ensemble_test_pred) ** 2))), 3),
            'MAPE_pct': round(calculate_mape(y_true, ensemble_test_pred), 3),
            'R_squared': round(float(1 - np.sum((y_true - ensemble_test_pred) ** 2) /
                                       np.sum((y_true - y_true.mean()) ** 2)), 4),
        }

    # 未来集成预测
    ensemble_future_pred = None
    if lstm_future_pred is not None and prophet_result.get('forecast') is not None and xgb_result.get('future_predictions'):
        # Prophet future: DataFrame with yhat
        prophet_future = prophet_result['forecast']['yhat'].values[-forecast_months:]
        xgb_future = [p['predicted_ppi'] for p in xgb_result['future_predictions']]
        lstm_future = [p['predicted_ppi'] for p in lstm_future_pred]
        future_dates = pd.date_range(df['date'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
        ensemble_vals = (
            weights['xgboost'] * np.array(xgb_future) +
            weights['prophet'] * np.array(prophet_future) +
            weights['lstm'] * np.array(lstm_future)
        )
        ensemble_future_pred = [
            {'date': d, 'predicted_ppi': float(v)}
            for d, v in zip(future_dates, ensemble_vals)
        ]

    return {
        'status': 'success',
        'lstm_params': lstm_params,
        'weights': weights,
        'individual_mapes': mapes,
        'xgboost_metrics': xgb_result.get('metrics') if xgb_result.get('status') == 'success' else None,
        'prophet_metrics': prophet_result.get('metrics') if prophet_result.get('status') == 'success' else None,
        'lstm_metrics': lstm_result.get('metrics') if lstm_result.get('status') == 'success' else None,
        'ensemble_metrics': ensemble_metrics,
        'ensemble_future_predictions': ensemble_future_pred,
        'xgboost_future': xgb_result.get('future_predictions') if xgb_result.get('status') == 'success' else None,
        'prophet_future': prophet_result.get('forecast') if prophet_result.get('status') == 'success' else None,
        'lstm_future': lstm_future_pred,
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ppi_monthly import load_monthly_ppi
    df = load_monthly_ppi()
    print(f'数据: {len(df)} 月')
    result = train_ensemble(df, test_months=24, forecast_months=12)
    print()
    print('=== 单一模型 vs 集成模型评估对比 ===')
    for m in ['xgboost', 'prophet', 'lstm', 'ensemble']:
        key = f'{m}_metrics'
        if key in result and result[key]:
            met = result[key]
            print(f"{m.upper():<12} MAE={met.get('MAE'):<6} RMSE={met.get('RMSE'):<6} MAPE%={met.get('MAPE_pct'):<6} R²={met.get('R_squared')}")
    print()
    print(f'权重: {result["weights"]}')
