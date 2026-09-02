"""
月度 PPI 时间序列预测模块（PyTorch 版）
功能：用 akshare 抓取的 132 个月度真实数据点（2015-2025）训练 Prophet + XGBoost + LSTM（PyTorch）三模型对比
作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目

数据来源：akshare.macro_china_ppi() 间接从国家统计局月度发布的 PPI 总指数
样本规模：132 月度点
窗口设置：seq_length=12（月度季节性捕获）
训练/测试划分：2015-2023 训练（108 点），2024-2025 测试（24 点）
"""

import pandas as pd
import numpy as np
from typing import Dict
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
    """月度数据特征工程"""
    df = df.sort_values('date').reset_index(drop=True).copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    df['yoy_change'] = df[target_col] - df[target_col].shift(12)
    df['mom_change'] = df[target_col] - df[target_col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df


def get_monthly_feature_columns() -> list:
    return [
        'year', 'month', 'quarter',
        'lag_1', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
        'yoy_change', 'mom_change',
    ]


def train_prophet_monthly(df: pd.DataFrame, target_col: str = 'ppi_index', forecast_months: int = 12) -> Dict:
    """Prophet 月度预测"""
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


def train_xgboost_monthly(df_features, feature_cols, target_col='ppi_index', test_months=24, forecast_months=12) -> Dict:
    """XGBoost 月度预测"""
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
        model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred_test)
        y_pred_train = model.predict(X_train)
        train_metrics = calculate_metrics(y_train.values, y_pred_train)
        feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        future_predictions = []
        last_row = df_features.iloc[-1].copy()
        history = df_features[[target_col]].copy()
        for i in range(forecast_months):
            next_date = last_row['date'] + pd.DateOffset(months=1)
            next_year = next_date.year
            next_month = next_date.month
            next_quarter = (next_month - 1) // 3 + 1
            feat = {
                'year': next_year, 'month': next_month, 'quarter': next_quarter,
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
            future_predictions.append({'date': next_date, 'predicted_ppi': pred})
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


def train_lstm_pytorch(
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    seq_length: int = 12,
    test_months: int = 24,
    forecast_months: int = 12,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 100,
    lr: float = 0.001,
) -> Dict:
    """LSTM 月度预测（PyTorch 版）"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {'status': 'error', 'reason': 'torch not installed'}
    if len(df) < seq_length + test_months + 1:
        return {'status': 'error', 'reason': f'数据不足（需要至少 {seq_length + test_months + 1} 个月）'}
    try:
        # 设种子保证可复现
        torch.manual_seed(42)
        np.random.seed(42)

        values = df[target_col].values.astype(float)
        split_idx = len(values) - test_months
        train_values = values[:split_idx]
        test_values = values[split_idx:]

        # 标准化
        mean = train_values.mean()
        std = train_values.std()
        if std == 0:
            std = 1
        train_scaled = (train_values - mean) / std
        test_scaled = (test_values - mean) / std

        # 构造序列
        def make_seq(data):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        X_train_seq, y_train_seq = make_seq(train_scaled)
        X_test_seq, y_test_seq = make_seq(test_scaled)
        X_train_t = torch.from_numpy(X_train_seq).unsqueeze(-1)  # (n, seq, 1)
        y_train_t = torch.from_numpy(y_train_seq).unsqueeze(-1)  # (n, 1)
        X_test_t = torch.from_numpy(X_test_seq).unsqueeze(-1)

        # 模型定义
        class LSTMModel(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout):
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
                last = lstm_out[:, -1, :]  # (batch, hidden)
                return self.fc(last)

        model = LSTMModel(hidden_size, num_layers, dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 训练 + Early Stopping
        best_loss = float('inf')
        best_state = None
        patience = 15
        no_improve = 0
        batch_size = 8
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

        # 测试集评估
        model.eval()
        with torch.no_grad():
            y_pred_test_scaled = model(X_test_t).numpy().flatten()
        y_pred_test = y_pred_test_scaled * std + mean
        y_true_test = test_values[seq_length:]
        metrics = calculate_metrics(y_true_test, y_pred_test)
        # 训练集评估
        with torch.no_grad():
            y_pred_train_scaled = model(X_train_t).numpy().flatten()
        y_pred_train = y_pred_train_scaled * std + mean
        y_true_train = train_values[seq_length:]
        train_metrics = calculate_metrics(y_true_train, y_pred_train)

        # 递归预测未来 forecast_months
        future_predictions = []
        last_seq = ((values[-seq_length:] - mean) / std).astype(np.float32)
        current_seq = torch.from_numpy(last_seq).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        with torch.no_grad():
            for _ in range(forecast_months):
                pred_scaled = model(current_seq).item()
                pred = pred_scaled * std + mean
                next_date = df['date'].iloc[-1] + pd.DateOffset(months=len(future_predictions) + 1)
                future_predictions.append({'date': next_date, 'predicted_ppi': pred})
                # 滚动窗口
                new_seq = torch.cat([current_seq[:, 1:, :], torch.tensor([[[pred_scaled]]], dtype=torch.float32)], dim=1)
                current_seq = new_seq

        return {
            'status': 'success',
            'metrics': metrics,
            'train_metrics': train_metrics,
            'future_predictions': future_predictions,
        }
    except Exception as e:
        logger.error(f'LSTM PyTorch 训练失败: {e}')
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'reason': str(e)}


def train_all_monthly_models(
    df: pd.DataFrame,
    target_col: str = 'ppi_index',
    test_months: int = 24,
    forecast_months: int = 12,
    lstm_params: dict = None,
) -> Dict:
    """
    训练月度三模型（Prophet + XGBoost + LSTM PyTorch）+ 评估对比
    lstm_params: {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'seq_length': 12, 'epochs': 100}
    """
    if df.empty:
        return {'status': 'error', 'reason': '输入数据为空'}
    if lstm_params is None:
        lstm_params = {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'seq_length': 12, 'epochs': 100}

    result = {
        'status': 'success',
        'total_points': len(df),
        'test_months': test_months,
        'forecast_months': forecast_months,
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d'),
        },
        'lstm_params': lstm_params,
    }

    # Prophet
    prophet_result = train_prophet_monthly(df, target_col, forecast_months)
    if prophet_result.get('status') == 'success':
        result['prophet'] = {
            'metrics': prophet_result['metrics'],
            'forecast': prophet_result['forecast'],
            'historical_fit': prophet_result['historical_fit'],
        }

    # XGBoost
    df_features = build_features_monthly(df, target_col)
    feature_cols = get_monthly_feature_columns()
    xgb_result = train_xgboost_monthly(df_features, feature_cols, target_col, test_months, forecast_months)
    if xgb_result.get('status') == 'success':
        result['xgboost'] = {
            'metrics': xgb_result['metrics'],
            'train_metrics': xgb_result['train_metrics'],
            'feature_importance': xgb_result['feature_importance'].to_dict(),
            'future_predictions': xgb_result['future_predictions'],
        }

    # LSTM（PyTorch）
    lstm_result = train_lstm_pytorch(
        df, target_col,
        seq_length=lstm_params.get('seq_length', 12),
        test_months=test_months,
        forecast_months=forecast_months,
        hidden_size=lstm_params.get('hidden_size', 64),
        num_layers=lstm_params.get('num_layers', 2),
        dropout=lstm_params.get('dropout', 0.1),
        epochs=lstm_params.get('epochs', 100),
        lr=lstm_params.get('lr', 0.001),
    )
    if lstm_result.get('status') == 'success':
        result['lstm'] = {
            'metrics': lstm_result['metrics'],
            'train_metrics': lstm_result['train_metrics'],
            'future_predictions': lstm_result['future_predictions'],
        }
    else:
        logger.warning(f'LSTM 训练失败: {lstm_result.get("reason")}')

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
        print(f'数据: {len(df)} 月')
        result = train_all_monthly_models(df, test_months=24, forecast_months=12)
        print('\n=== 月度三模型评估对比 ===')
        for m in ['prophet', 'xgboost', 'lstm']:
            if m in result:
                met = result[m]['metrics']
                print(f"{m.upper():<10} MAE={met['MAE']:<6} RMSE={met['RMSE']:<6} MAPE%={met['MAPE_pct']:<6} R²={met['R_squared']}")
            else:
                print(f'{m.upper():<10} 失败')
