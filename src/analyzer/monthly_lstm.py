"""
月度 PPI 时间序列预测模块
功能：用 akshare 抓取的 132 个月度真实数据点（2015-2025）训练 LSTM + Prophet + XGBoost 三模型对比
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

数据来源：akshare.macro_china_ppi() 间接从国家统计局月度发布的 PPI 总指数
样本规模：132 月度点（远超 LSTM 阈值）
窗口设置：seq_length=12（月度季节性捕获）
训练/测试划分：2015-2023 训练（108 点），2024-2025 测试（24 点）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """计算模型评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE_pct': np.nan, 'R_squared': np.nan}
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if y_true.mean() > 0 else 0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {
        'MAE': round(float(mae), 3),
        'RMSE': round(float(rmse), 3),
        'MAPE_pct': round(float(mape), 3),
        'R_squared': round(float(r_squared), 4),
    }


def build_features_monthly(df: pd.DataFrame, target_col: str = 'ppi_index') -> pd.DataFrame:
    """
    月度数据特征工程
    - 滞后特征（lag1, lag3, lag6, lag12）
    - 滚动均值（3 月 / 6 月 / 12 月）
    - 时间特征（年 / 月 / 季度 / 是否年初）
    """
    df = df.sort_values('date').reset_index(drop=True).copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    # 滞后特征
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    # 滚动统计
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    # 同比（去年同期）
    df['yoy_change'] = df[target_col] - df[target_col].shift(12)
    # 环比（上一月）
    df['mom_change'] = df[target_col] - df[target_col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df


def get_monthly_feature_columns() -> list:
    """月度特征列名清单"""
    return [
        'year', 'month', 'quarter',
        'lag_1', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
        'yoy_change', 'mom_change',
    ]


def train_prophet_monthly(
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    forecast_months: int = 12,
) -> Dict:
    """
    Prophet 月度预测（最擅长月度时间序列）
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {'status': 'error', 'reason': 'prophet not installed'}
    if len(df) < 24:
        return {'status': 'error', 'reason': '数据不足 24 个月'}
    prophet_df = df[['date', target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95,
            changepoint_prior_scale=0.05,
        )
        model.fit(prophet_df)
        historical = model.predict(prophet_df[['ds']])
        metrics = calculate_metrics(prophet_df['y'].values, historical['yhat'].values)
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast = model.predict(future)
        return {
            'status': 'success',
            'model': model,
            'historical_fit': historical,
            'forecast': forecast.tail(forecast_months),
            'metrics': metrics,
        }
    except Exception as e:
        logger.error(f'Prophet 月度训练失败: {e}')
        return {'status': 'error', 'reason': str(e)}


def train_xgboost_monthly(
    df_features: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'ppi_index',
    test_months: int = 24,
    forecast_months: int = 12,
) -> Dict:
    """
    XGBoost 月度预测（基于特征工程）
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {'status': 'error', 'reason': 'xgboost not installed'}
    if len(df_features) < test_months + 12:
        return {'status': 'error', 'reason': '训练数据不足'}
    try:
        split_idx = len(df_features) - test_months
        train = df_features.iloc[:split_idx]
        test = df_features.iloc[split_idx:]
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred_test)
        y_pred_train = model.predict(X_train)
        train_metrics = calculate_metrics(y_train.values, y_pred_train)
        feature_importance = pd.Series(
            model.feature_importances_,
            index=feature_cols,
        ).sort_values(ascending=False)
        # 递归预测未来 forecast_months
        future_predictions = []
        last_row = df_features.iloc[-1].copy()
        history = df_features[[target_col]].copy()
        for i in range(forecast_months):
            next_date = last_row['date'] + pd.DateOffset(months=1)
            next_year = next_date.year
            next_month = next_date.month
            next_quarter = (next_month - 1) // 3 + 1
            # 构造特征
            feat = {
                'year': next_year,
                'month': next_month,
                'quarter': next_quarter,
            }
            for lag in [1, 3, 6, 12]:
                feat[f'lag_{lag}'] = history[target_col].iloc[-lag] if len(history) >= lag else history[target_col].iloc[-1]
            for window in [3, 6, 12]:
                feat[f'rolling_mean_{window}'] = history[target_col].iloc[-window:].mean()
                feat[f'rolling_std_{window}'] = history[target_col].iloc[-window:].std()
            feat['yoy_change'] = history[target_col].iloc[-1] - history[target_col].iloc[-13] if len(history) >= 13 else 0
            feat['mom_change'] = history[target_col].iloc[-1] - history[target_col].iloc[-2] if len(history) >= 2 else 0
            X_future = pd.DataFrame([feat])[feature_cols]
            pred = float(model.predict(X_future)[0])
            future_predictions.append({
                'date': next_date,
                'predicted_ppi': pred,
            })
            history = pd.concat([history, pd.DataFrame({target_col: [pred]})], ignore_index=True)
        return {
            'status': 'success',
            'model': model,
            'metrics': metrics,
            'train_metrics': train_metrics,
            'feature_importance': feature_importance,
            'future_predictions': future_predictions,
            'test_predictions': y_pred_test,
            'test_actual': y_test.values,
        }
    except Exception as e:
        logger.error(f'XGBoost 月度训练失败: {e}')
        return {'status': 'error', 'reason': str(e)}


def train_lstm_monthly(
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    seq_length: int = 12,
    test_months: int = 24,
    forecast_months: int = 12,
    epochs: int = 50,
) -> Dict:
    """
    LSTM 月度预测（用 132 月度点真训练，窗口 12 月捕获季节性）
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import StandardScaler
        # 抑制 TF 日志
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        return {'status': 'error', 'reason': 'tensorflow not installed'}
    if len(df) < seq_length + test_months + 1:
        return {'status': 'error', 'reason': f'数据不足（需要至少 {seq_length + test_months + 1} 个月）'}
    try:
        values = df[target_col].values.astype(float)
        # 划分
        split_idx = len(values) - test_months
        train_values = values[:split_idx]
        test_values = values[split_idx:]
        # 归一化
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
        # 构造序列（返回 3D：[样本数, seq_length, 1] 适配 LSTM）
        def make_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len].reshape(-1, 1))  # 每步 reshape 成 (seq_len, 1)
                y.append(data[i + seq_len])
            return np.array(X), np.array(y)
        X_train_seq, y_train_seq = make_sequences(train_scaled, seq_length)
        X_test_seq, y_test_seq = make_sequences(test_scaled, seq_length)
        # LSTM 模型（用 Input 层显式声明避免 TF 2.20 unknown rank 错误）
        from tensorflow.keras.layers import Input
        model = Sequential([
            Input(shape=(seq_length, 1)),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=8, verbose=0, validation_split=0.1)
        # 测试集评估
        y_pred_test_scaled = model.predict(X_test_seq, verbose=0).flatten()
        y_pred_test = scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        y_true_test = test_values[seq_length:]
        metrics = calculate_metrics(y_true_test, y_pred_test)
        # 训练集评估
        y_pred_train_scaled = model.predict(X_train_seq, verbose=0).flatten()
        y_pred_train = scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
        y_true_train = train_values[seq_length:]
        train_metrics = calculate_metrics(y_true_train, y_pred_train)
        # 递归预测未来 forecast_months
        future_predictions = []
        # 用最后 seq_length 个点（含测试集末尾）作为初始窗口
        last_seq = scaler.transform(values[-seq_length:].reshape(-1, 1)).flatten()
        current_seq = last_seq.copy().reshape(-1, 1)
        for _ in range(forecast_months):
            pred_scaled = model.predict(
                current_seq.reshape(1, seq_length, 1), verbose=0
            )[0][0]
            pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            next_date = df['date'].iloc[-1] + pd.DateOffset(months=len(future_predictions) + 1)
            future_predictions.append({
                'date': next_date,
                'predicted_ppi': pred,
            })
            current_seq = np.append(current_seq[1:], pred_scaled)
        return {
            'status': 'success',
            'model': model,
            'metrics': metrics,
            'train_metrics': train_metrics,
            'future_predictions': future_predictions,
            'test_predictions': y_pred_test,
            'test_actual': y_true_test,
            'test_dates': df['date'].iloc[split_idx + seq_length:].values,
        }
    except Exception as e:
        logger.error(f'LSTM 月度训练失败: {e}')
        return {'status': 'error', 'reason': str(e)}


def train_all_monthly_models(
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    test_months: int = 24,
    forecast_months: int = 12,
) -> Dict:
    """
    训练月度三模型（Prophet + XGBoost + LSTM）+ 评估对比
    """
    if df.empty:
        return {'status': 'error', 'reason': '输入数据为空'}
    result = {
        'status': 'success',
        'total_points': len(df),
        'test_months': test_months,
        'forecast_months': forecast_months,
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d'),
        },
    }
    # Prophet
    prophet_result = train_prophet_monthly(df, target_col, forecast_months)
    if prophet_result.get('status') == 'success':
        result['prophet'] = {
            'metrics': prophet_result['metrics'],
            'forecast': prophet_result['forecast'],
            'historical_fit': prophet_result['historical_fit'],
        }
    # XGBoost（需要先特征工程）
    df_features = build_features_monthly(df, target_col)
    feature_cols = get_monthly_feature_columns()
    xgb_result = train_xgboost_monthly(
        df_features, feature_cols, target_col, test_months, forecast_months
    )
    if xgb_result.get('status') == 'success':
        result['xgboost'] = {
            'metrics': xgb_result['metrics'],
            'train_metrics': xgb_result['train_metrics'],
            'feature_importance': xgb_result['feature_importance'].to_dict(),
            'future_predictions': xgb_result['future_predictions'],
        }
    # LSTM
    lstm_result = train_lstm_monthly(
        df, target_col, seq_length=12, test_months=test_months,
        forecast_months=forecast_months, epochs=50,
    )
    if lstm_result.get('status') == 'success':
        result['lstm'] = {
            'metrics': lstm_result['metrics'],
            'train_metrics': lstm_result['train_metrics'],
            'future_predictions': lstm_result['future_predictions'],
        }
    return result


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ppi_monthly import load_monthly_ppi

    df = load_monthly_ppi()
    if df.empty:
        print('没有月度数据')
    else:
        print(f'\n数据: {len(df)} 个月度点')
        result = train_all_monthly_models(df, test_months=24, forecast_months=12)
        print('\n=== 月度训练结果 ===')
        if 'prophet' in result:
            print(f"Prophet 评估: {result['prophet']['metrics']}")
        if 'xgboost' in result:
            print(f"XGBoost 评估: {result['xgboost']['metrics']}")
        if 'lstm' in result:
            print(f"LSTM 评估: {result['lstm']['metrics']}")
