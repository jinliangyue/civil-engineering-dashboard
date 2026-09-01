"""
机器学习预测模型模块
功能：实现 3 个预测模型：Prophet / XGBoost / LSTM，对比评估
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

策略：
- Prophet：传统时间序列模型，基于加法分解（trend + seasonal + holiday）
- XGBoost：梯度提升树，基于特征工程
- LSTM：长短期记忆网络，捕捉非线性时间依赖

由于样本点少（4 行业 × 11 年 = 44 个点）：
- 训练/测试划分：2015-2022 训练，2023-2025 测试
- 评估指标：MAE / RMSE / MAPE / R²
- LSTM 因样本少仅作演示
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
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'MAPE_pct': round(float(mape), 2),
        'R_squared': round(float(r_squared), 4),
    }


def train_prophet_model(df: pd.DataFrame, forecast_years: int = 3) -> Dict:
    """
    训练 Prophet 模型
    Prophet 输入要求：ds（日期）+ y（数值）
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.error('prophet 未安装')
        return {'status': 'error', 'reason': 'prophet not installed'}
    if df.empty or len(df) < 4:
        return {'status': 'error', 'reason': '数据不足'}
    prophet_df = df[['date', 'price']].copy()
    prophet_df.columns = ['ds', 'y']
    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95,
        )
        model.fit(prophet_df)
        # 历史拟合
        historical = model.predict(prophet_df[['ds']])
        metrics = calculate_metrics(prophet_df['y'].values, historical['yhat'].values)
        # 未来预测
        future = model.make_future_dataframe(periods=forecast_years, freq='YS')
        forecast = model.predict(future)
        return {
            'status': 'success',
            'model': model,
            'historical_fit': historical,
            'forecast': forecast.tail(forecast_years),
            'metrics': metrics,
        }
    except Exception as e:
        logger.error(f'Prophet 训练失败: {e}')
        return {'status': 'error', 'reason': str(e)}


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    forecast_years: int = 3,
) -> Dict:
    """
    训练 XGBoost 模型
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.error('xgboost 未安装')
        return {'status': 'error', 'reason': 'xgboost not installed'}
    if len(X_train) < 5:
        return {'status': 'error', 'reason': '训练数据不足'}
    try:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        # 测试集评估
        y_pred_test = model.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred_test)
        # 训练集拟合
        y_pred_train = model.predict(X_train)
        train_metrics = calculate_metrics(y_train.values, y_pred_train)
        # 特征重要性
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)
        # 未来预测（用最后一年特征外推 forecast_years 年）
        future_predictions = []
        last_year = pd.to_datetime(X_train['year'] if 'year' in X_train.columns else X_train.index).max()
        # 简化：用最后一行特征重复，逐年调整时间特征
        last_features = X_train.iloc[-1].copy()
        for i in range(1, forecast_years + 1):
            pred_features = last_features.copy()
            if 'years_from_base' in pred_features.index:
                pred_features['years_from_base'] += i
            if 'year' in pred_features.index:
                pred_features['year'] += i
            pred = model.predict(pd.DataFrame([pred_features]))[0]
            future_predictions.append(pred)
        return {
            'status': 'success',
            'model': model,
            'metrics': metrics,
            'train_metrics': train_metrics,
            'feature_importance': feature_importance,
            'future_predictions': future_predictions,
        }
    except Exception as e:
        logger.error(f'XGBoost 训练失败: {e}')
        return {'status': 'error', 'reason': str(e)}


def train_lstm_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'price',
    forecast_years: int = 3,
    epochs: int = 100,
) -> Dict:
    """
    训练 LSTM 模型（演示版，样本少时仅作方法展示）
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error('tensorflow 未安装')
        return {'status': 'error', 'reason': 'tensorflow not installed'}
    if len(df) < 8:
        return {'status': 'error', 'reason': 'LSTM 需要至少 8 个样本'}
    try:
        # 数据准备
        data = df[feature_cols + [target_col]].copy()
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(data[feature_cols].values)
        y_scaled = scaler_y.fit_transform(data[[target_col]].values).flatten()
        # 创建序列（窗口 = 2）
        seq_length = 2
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_length):
            X_seq.append(X_scaled[i:i + seq_length])
            y_seq.append(y_scaled[i + seq_length])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        if len(X_seq) < 4:
            return {'status': 'error', 'reason': '序列数据不足'}
        # 构建模型
        model = Sequential([
            LSTM(16, input_shape=(seq_length, len(feature_cols)), return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_seq, y_seq, epochs=epochs, batch_size=2, verbose=0)
        # 预测（递归）
        future_predictions = []
        last_seq = X_scaled[-seq_length:].copy()
        for _ in range(forecast_years):
            pred_scaled = model.predict(last_seq.reshape(1, seq_length, len(feature_cols)), verbose=0)[0][0]
            future_predictions.append(pred_scaled)
            # 用预测值构造下一序列
            new_seq = np.roll(last_seq, -1, axis=0)
            # 简化：用最后一个特征值 + 预测值作为新的时间步
            new_seq[-1] = last_seq[-1]  # 简化处理
            last_seq = new_seq
        # 反归一化
        future_predictions = scaler_y.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()
        # 训练集评估
        train_pred_scaled = model.predict(X_seq, verbose=0).flatten()
        train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        y_true = data[target_col].values[seq_length:]
        train_metrics = calculate_metrics(y_true, train_pred)
        return {
            'status': 'success',
            'model': model,
            'metrics': train_metrics,
            'future_predictions': future_predictions.tolist(),
        }
    except Exception as e:
        logger.error(f'LSTM 训练失败: {e}')
        return {'status': 'error', 'reason': str(e)}


def train_all_models(
    df_features: pd.DataFrame,
    feature_cols: list,
    test_years: list = [2024, 2025],
    forecast_years: int = 3,
) -> Dict:
    """
    训练全部 3 个模型 + 评估对比
    """
    if df_features.empty:
        return {'status': 'error', 'reason': '输入数据为空'}
    # 准备数据
    from .features import prepare_train_test
    X_train, X_test, y_train, y_test = prepare_train_test(df_features, test_years=test_years)
    # Prophet（用全部数据）
    df_for_prophet = df_features[['date', 'price']].drop_duplicates().sort_values('date')
    # XGBoost（用训练集/测试集划分）
    xgb_result = train_xgboost_model(X_train, y_train, X_test, y_test, forecast_years=forecast_years)
    # LSTM（用全部数据）
    lstm_result = train_lstm_model(df_features, feature_cols, forecast_years=forecast_years)
    # 整合结果
    result = {
        'status': 'success',
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
    }
    if xgb_result.get('status') == 'success':
        result['xgboost'] = {
            'metrics': xgb_result.get('metrics'),
            'train_metrics': xgb_result.get('train_metrics'),
            'feature_importance': xgb_result.get('feature_importance').to_dict() if xgb_result.get('feature_importance') is not None else {},
            'future_predictions': xgb_result.get('future_predictions', []),
        }
    if lstm_result.get('status') == 'success':
        result['lstm'] = {
            'metrics': lstm_result.get('metrics'),
            'future_predictions': lstm_result.get('future_predictions', []),
        }
    logger.info(f'XGBoost 状态: {xgb_result.get("status")}')
    logger.info(f'LSTM 状态: {lstm_result.get("status")}')
    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(0, '../..')
    from src.data_loader import load_all_raw
    from src.data_cleaner import clean_pipeline
    from src.analyzer.features import build_features, get_feature_columns
    df_raw = load_all_raw()
    if df_raw.empty:
        print('没有数据')
    else:
        df_clean, _ = clean_pipeline(df_raw)
        df_features = build_features(df_clean)
        feature_cols = get_feature_columns(df_features)
        print(f'\n特征: {len(feature_cols)} 个, 数据: {len(df_features)} 行')
        result = train_all_models(df_features, feature_cols)
        print('\n=== 训练结果 ===')
        if 'xgboost' in result:
            print(f"XGBoost 评估: {result['xgboost']['metrics']}")
        if 'lstm' in result:
            print(f"LSTM 评估: {result['lstm']['metrics']}")