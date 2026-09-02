"""
月度 PPI 模型训练与预测模块（Phase 2 v3.1 严格版）

数据边界（绝对遵守）：
- 完整数据：2015-01 ~ 2025-12 = 132 点（akshare.macro_china_ppi 间接抓取）
- Train + Validation：2015-01 ~ 2023-12 = 108 点
- Final Test：2024-01 ~ 2025-12 = 24 点（只用一次）

每个模型的训练 / 预测方式：

Prophet
  fit(train_val_df)            ← 108 点
  predict(test_df)            ← 24 点 out-of-sample
  evaluate(actuals, preds)    ← Test 真实值 vs 预测值

XGBoost
  build_features on train_val + test
  fit on train_val features
  Test: rolling one-step-ahead
    每个 test 时刻 t：
      feat = history[-seq:] (real actuals)
      pred = model.predict(feat)
      history.append(test_actual[t])

LSTM
  scaler.fit(train_val_df) ONLY ← 不接触 test
  fit on train+val sequences with best_params (from P0.3)
  Test: rolling one-step-ahead
  Future: recursive (12 步，无 ground truth)

Future Forecast（不可评估）
  XGBoost / LSTM 递归预测 2026-01 ~ 2026-12
  仅供 Streamlit 展示，不计算 MAPE

重要修复（P0.4 vs 旧版）：
1. 修复 build_features_monthly 的 target leakage
   - 旧: rolling(window).mean() 包含 target 当前值
   - 新: rolling(shift(1)).mean() 只用过去真实值
2. 修复 Prophet in-sample 评估
   - 旧: fit 全 132 点 + 同一份数据 predict
   - 新: fit train+val(108) + predict test(24) out-of-sample
3. 修复 XGBoost Test 评估
   - 旧: 直接 iloc 切 test features predict
   - 新: rolling one-step-ahead，每步用截至当前点的真实历史
4. LSTM 接 P0.3 best_params + fit train+val (108)
5. 强制 boundary assertions：传入数据长度 / 时间范围严格校验

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === 数据边界常量（Phase 2 v3.1 冻结）===
TRAIN_END_DATE = pd.Timestamp('2023-12-31')
TEST_START_DATE = pd.Timestamp('2024-01-01')
EXPECTED_TRAIN_VAL_LEN = 108
EXPECTED_TEST_LEN = 24
EXPECTED_TOTAL_LEN = 132


# === Boundary Assertions ===
def _verify_data_boundary(train_val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    严格验证传入的 train_val_df 和 test_df 符合 108/24 边界

    Raises:
        AssertionError: 任一边界不满足
    """
    assert len(train_val_df) == EXPECTED_TRAIN_VAL_LEN, \
        f"train_val_df 长度 = {len(train_val_df)} != {EXPECTED_TRAIN_VAL_LEN}（应为 Train + Validation = 108 月）"
    assert len(test_df) == EXPECTED_TEST_LEN, \
        f"test_df 长度 = {len(test_df)} != {EXPECTED_TEST_LEN}（应为 Final Test = 24 月）"

    train_end = train_val_df['date'].max()
    test_start = test_df['date'].min()
    test_end = test_df['date'].max()

    assert train_end <= TRAIN_END_DATE, \
        f"train_val_df 最末日期 {train_end} > {TRAIN_END_DATE}（包含 Final Test 段 → 数据泄漏！）"
    assert test_start >= TEST_START_DATE, \
        f"test_df 起始日期 {test_start} < {TEST_START_DATE}（包含 Train/Val 段 → 数据泄漏！）"

    # 关键：train_val_df 不应包含 test 段
    assert (train_val_df['date'] < test_start).all(), \
        "train_val_df 包含 test 段数据 → 数据泄漏！"
    assert (test_df['date'] >= test_start).all(), \
        "test_df 包含 train_val 段数据 → 数据泄漏！"

    total = len(train_val_df) + len(test_df)
    assert total == EXPECTED_TOTAL_LEN, \
        f"总数据点 {total} != {EXPECTED_TOTAL_LEN}"

    logger.info(
        f"=== 数据边界验证 ===\n"
        f"  train_val: {len(train_val_df)} 月 ({train_val_df['date'].min().strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')})\n"
        f"  test:      {len(test_df)} 月 ({test_start.strftime('%Y-%m')} ~ {test_end.strftime('%Y-%m')})\n"
        f"  total:     {total} 月"
    )


# === 修复后的 Feature Engineering（严格 causal）===
def build_features_monthly_causal(df: pd.DataFrame, target_col: str = 'ppi_index') -> pd.DataFrame:
    """
    构建严格 causal 的月度特征

    关键修复 vs 旧版本：
    - 所有 rolling features 必须基于 shift(1) — 只用过去真实值
    - yoy_change / mom_change 也基于 shift(1)
    - 绝对不允许使用 target 当前值或未来值构造特征

    Args:
        df: 完整月度 DataFrame（含 date + target_col）
        target_col: 目标列名

    Returns:
        带特征的 DataFrame（已 dropna）
    """
    df = df.sort_values('date').reset_index(drop=True).copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Lag features（shift N 本身只用过去值 → ✓ OK）
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling features：必须 shift(1) 后再 rolling
    # 否则 rolling_mean_3 在 t 处 = mean(y[t-2], y[t-1], y[t]) 含当前 target → leakage
    target_shifted_1 = df[target_col].shift(1)
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = target_shifted_1.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = target_shifted_1.rolling(window=window).std()

    # yoy_change / mom_change：必须 shift(1)
    # 旧: df[target_col] - df[target_col].shift(12) 在 t 处含 y[t]
    # 新: df[target_col].shift(1) - df[target_col].shift(13)
    df['yoy_change'] = df[target_col].shift(1) - df[target_col].shift(13)
    df['mom_change'] = df[target_col].shift(1) - df[target_col].shift(2)

    return df.dropna().reset_index(drop=True)


def get_monthly_feature_columns() -> list:
    """月度特征列名清单"""
    return [
        'year', 'month', 'quarter',
        'lag_1', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
        'yoy_change', 'mom_change',
    ]


# === Prophet ===
def train_prophet_final(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'ppi_index',
) -> Tuple[Dict, np.ndarray, object]:
    """
    Prophet Final: fit(train+val=108) → predict test(24) out-of-sample

    Returns:
        (metrics dict, test_pred array, fitted Prophet model)
    """
    from prophet import Prophet

    _verify_data_boundary(train_val_df, test_df)

    prophet_df = train_val_df[['date', target_col]].copy()
    prophet_df.columns = ['ds', 'y']

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    # Predict only test（out-of-sample）
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    forecast = model.predict(future)
    test_pred = forecast.tail(len(test_df))['yhat'].values

    test_actuals = test_df[target_col].values

    from src.evaluation.metrics import mape, mae, rmse, r2
    metrics = {
        'MAE': mae(test_actuals, test_pred),
        'RMSE': rmse(test_actuals, test_pred),
        'MAPE_pct': mape(test_actuals, test_pred),
        'R_squared': r2(test_actuals, test_pred),
    }

    logger.info(
        f"Prophet Test (out-of-sample): MAE={metrics['MAE']:.4f} "
        f"RMSE={metrics['RMSE']:.4f} MAPE={metrics['MAPE_pct']:.4f}% R²={metrics['R_squared']:.4f}"
    )
    return metrics, test_pred, model


def predict_prophet_future(prophet_model, horizon: int = 12) -> List[Dict]:
    """
    Prophet Future Forecast（仅展示，不评估）

    Args:
        prophet_model: 已 fit 在 train+val 上的 Prophet
        horizon: 预测未来月数（默认 12）

    Returns:
        [{'date': ..., 'predicted_ppi': ...}, ...]
    """
    future = prophet_model.make_future_dataframe(periods=horizon, freq='MS')
    forecast = prophet_model.predict(future)
    last_train_date = forecast['ds'].iloc[-(horizon + 1)]
    future_only = forecast[forecast['ds'] > last_train_date][['ds', 'yhat']]
    return [
        {'date': row['ds'], 'predicted_ppi': float(row['yhat'])}
        for _, row in future_only.iterrows()
    ]


# === XGBoost ===
def _construct_xgb_features_from_history(
    history: List[float],
    calendar_row: pd.Series,
) -> Dict:
    """
    根据 history（截至 t 之前的真实值列表）和 calendar_row（t 时刻的 year/month/quarter）
    构造 t 时刻的 XGBoost 特征 dict

    关键：所有 lag/rolling/derived features 都从 history 计算（不是从 calendar_row）
    calendar_row 只提供 year/month/quarter
    """
    feat = {
        'year': int(calendar_row['year']),
        'month': int(calendar_row['month']),
        'quarter': int(calendar_row['quarter']),
    }
    # Lag features
    for lag in [1, 3, 6, 12]:
        if len(history) >= lag:
            feat[f'lag_{lag}'] = history[-lag]
        else:
            feat[f'lag_{lag}'] = history[0]
    # Rolling features（基于 history）
    for window in [3, 6, 12]:
        if len(history) >= window:
            feat[f'rolling_mean_{window}'] = float(np.mean(history[-window:]))
            feat[f'rolling_std_{window}'] = float(np.std(history[-window:]))
        else:
            feat[f'rolling_mean_{window}'] = float(np.mean(history))
            feat[f'rolling_std_{window}'] = float(np.std(history))
    # yoy_change / mom_change（基于 history shift(1)）
    feat['yoy_change'] = history[-1] - history[-13] if len(history) >= 13 else 0.0
    feat['mom_change'] = history[-1] - history[-2] if len(history) >= 2 else 0.0
    return feat


def train_xgboost_final(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'ppi_index',
) -> Tuple[Dict, np.ndarray, object, List[float]]:
    """
    XGBoost Final: fit on train+val features → rolling one-step-ahead predict test

    Returns:
        (metrics dict, test_preds array, fitted XGBoost model, final history list)
    """
    from xgboost import XGBRegressor

    _verify_data_boundary(train_val_df, test_df)

    # 构造特征（在完整 df 上，但 build_features_monthly_causal 严格 causal）
    full_df = pd.concat([train_val_df, test_df]).sort_values('date').reset_index(drop=True)
    df_features = build_features_monthly_causal(full_df, target_col)

    train_end = train_val_df['date'].max()
    train_val_features = df_features[df_features['date'] <= train_end].copy()
    test_features = df_features[df_features['date'] > train_end].copy()

    feature_cols = get_monthly_feature_columns()

    X_train_val = train_val_features[feature_cols]
    y_train_val = train_val_features[target_col]

    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0,
    )
    model.fit(X_train_val, y_train_val)

    # === Rolling one-step-ahead Test 预测 ===
    # history 起始 = train_val_df 全部真实 ppi_index（不含 test 真实值）
    history = list(train_val_df[target_col].values)
    test_preds = []

    for idx in range(len(test_features)):
        row = test_features.iloc[idx]
        feat = _construct_xgb_features_from_history(history, row)
        feat_df = pd.DataFrame([feat])[feature_cols]
        pred = float(model.predict(feat_df)[0])
        test_preds.append(pred)
        # 关键：把该 test 时刻的真实值加入 history（用于下一步）
        history.append(test_features.iloc[idx][target_col])

    test_preds = np.array(test_preds)
    test_actuals = test_df[target_col].values

    from src.evaluation.metrics import mape, mae, rmse, r2
    metrics = {
        'MAE': mae(test_actuals, test_preds),
        'RMSE': rmse(test_actuals, test_preds),
        'MAPE_pct': mape(test_actuals, test_preds),
        'R_squared': r2(test_actuals, test_preds),
    }

    logger.info(
        f"XGBoost Test (rolling one-step-ahead, 24 steps): "
        f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
        f"MAPE={metrics['MAPE_pct']:.4f}% R²={metrics['R_squared']:.4f}"
    )
    return metrics, test_preds, model, history


def predict_xgboost_future_recursive(
    xgb_model: object,
    train_val_df: pd.DataFrame,
    feature_cols: list,
    horizon: int = 12,
    target_col: str = 'ppi_index',
) -> List[Dict]:
    """
    XGBoost Future Recursive Forecast（12 步递归，仅展示不评估）

    与 rolling one-step-ahead 不同：future 时没有真实值，lag 逐步变预测值
    """
    history = list(train_val_df[target_col].values)
    last_date = train_val_df['date'].max()
    future_preds = []

    for step in range(horizon):
        next_date = last_date + pd.DateOffset(months=step + 1)
        calendar_row = pd.Series({
            'year': next_date.year,
            'month': next_date.month,
            'quarter': (next_date.month - 1) // 3 + 1,
        })
        feat = _construct_xgb_features_from_history(history, calendar_row)
        feat_df = pd.DataFrame([feat])[feature_cols]
        pred = float(xgb_model.predict(feat_df)[0])
        future_preds.append({'date': next_date, 'predicted_ppi': pred})
        # 递归：history 追加预测值（无真实值）
        history.append(pred)

    return future_preds


# === LSTM ===
def train_lstm_final(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: Dict,
    target_col: str = 'ppi_index',
    epochs: int = 100,
) -> Tuple[Dict, np.ndarray, object, Tuple[float, float], np.ndarray]:
    """
    LSTM Final: fit on train+val(108) with P0.3 best_params → rolling one-step-ahead test(24)

    Args:
        train_val_df: Train + Validation (108 月)
        test_df: Final Test (24 月)
        best_params: P0.3 lstm_tuning.grid_search_lstm 输出 best_params
        target_col: 目标列
        epochs: 训练轮数

    Returns:
        (metrics dict, test_preds array, fitted LSTM model, (mean, std), test_actuals array)
    """
    import torch
    import torch.nn as nn

    _verify_data_boundary(train_val_df, test_df)

    seq_length = best_params['seq_length']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout = best_params['dropout']
    lr = best_params['lr']

    values_train_val = train_val_df[target_col].values.astype(float)
    values_test = test_df[target_col].values.astype(float)

    # === Scaler fit ONLY on train+val ===
    mean = float(values_train_val.mean())
    std = float(values_train_val.std())
    if std == 0:
        std = 1.0

    train_val_scaled = (values_train_val - mean) / std
    # Test 用 train+val 的 mean/std transform（transform 不算泄漏）
    test_scaled = (values_test - mean) / std

    # Train 序列（仅用 train+val）
    def make_seq_train(data):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].reshape(-1, 1))
            y.append(data[i + seq_length])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train_seq, y_train_seq = make_seq_train(train_val_scaled)

    # 模型定义
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 8), nn.ReLU(), nn.Linear(8, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train + Early Stopping on train loss
    X_train_t = torch.from_numpy(X_train_seq)
    y_train_t = torch.from_numpy(y_train_seq).unsqueeze(-1)

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

    # === Test 段 rolling one-step-ahead（完整 24 步覆盖 2024-01 ~ 2025-12）===
    # 关键修复：必须从 Test 第一个点（2024-01）开始预测，输出 24 个 predictions
    #
    # 正确逻辑：
    # - 初始 history = train_val_scaled（108 个 scaled 值，全部来自 Train+Val）
    # - 预测 2024-01: input = history[-6:] = 2023-07~2023-12（来自 Train+Val，合法）
    # - 把 2024-01 actual 加入 history
    # - 预测 2024-02: input = history[-6:] = 2023-08~2024-01（含 2024-01 actual，已发生）
    # - ...
    # - 预测 2025-12: input = history[-6:] = 2025-06~2025-11（不含 2025-12 actual）
    # - 最终 output 24 predictions
    #
    # 边界严格保证：
    # - 2024-01 actual 在预测 2024-01 时不可见（尚未发生）
    # - 2025-12 actual 在预测 2025-12 时不可见（预测完成前不可用）
    # - history 只能在当前点预测完成后追加 actual
    test_preds_scaled = []
    # history_scaled 初始 = train_val_scaled 全部 108 个 scaled 值
    history_scaled = list(train_val_scaled)

    for t in range(len(test_scaled)):
        # 取 history_scaled 最后 seq_length 个值
        if len(history_scaled) < seq_length:
            # Fallback（理论上不会触发，因为 train_val_scaled 有 108 个 > seq_length）
            input_seq = history_scaled[:seq_length]
        else:
            input_seq = history_scaled[-seq_length:]
        x = np.array(input_seq, dtype=np.float32).reshape(1, seq_length, 1)
        x_t = torch.from_numpy(x)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(x_t).numpy()[0][0]
        test_preds_scaled.append(pred_scaled)
        # 关键：把当前 test 点的真实值加入 history（用于下一步预测）
        history_scaled.append(test_scaled[t])

    # === 严格断言：24 个预测 ===
    assert len(test_preds_scaled) == 24, (
        f"LSTM Test predictions 数量 = {len(test_preds_scaled)} != 24。"
        f"必须严格覆盖 2024-01 ~ 2025-12 全部 24 个月。"
        f"Phase 2 v3.1 实验设计：24 consecutive rolling one-step-ahead predictions。"
    )

    test_preds_scaled = np.array(test_preds_scaled)
    # Inverse transform
    test_preds = test_preds_scaled * std + mean
    # 全部 24 个 actual（不再切片）
    test_actuals = values_test

    # === 进一步断言：predictions 和 actuals 长度一致 ===
    assert len(test_preds) == len(test_actuals) == 24, (
        f"len(preds)={len(test_preds)}, len(actuals)={len(test_actuals)}, 都不等于 24"
    )

    from src.evaluation.metrics import mape, mae, rmse, r2
    metrics = {
        'MAE': mae(test_actuals, test_preds),
        'RMSE': rmse(test_actuals, test_preds),
        'MAPE_pct': mape(test_actuals, test_preds),
        'R_squared': r2(test_actuals, test_preds),
    }

    logger.info(
        f"LSTM Test (rolling one-step-ahead, 完整 24 步, best_params from P0.3): "
        f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
        f"MAPE={metrics['MAPE_pct']:.4f}% R²={metrics['R_squared']:.4f} "
        f"n_pred={len(test_preds)}, n_actual={len(test_actuals)}"
    )
    return metrics, test_preds, model, (mean, std), test_actuals


def predict_lstm_future_recursive(
    lstm_model: object,
    train_val_df: pd.DataFrame,
    mean_std: Tuple[float, float],
    best_params: Dict,
    horizon: int = 12,
    target_col: str = 'ppi_index',
) -> List[Dict]:
    """
    LSTM Future Recursive Forecast（12 步递归，仅展示不评估）
    """
    import torch

    mean, std = mean_std
    seq_length = best_params['seq_length']

    last_seq = ((train_val_df[target_col].values[-seq_length:] - mean) / std).astype(np.float32)
    current_seq = torch.from_numpy(last_seq).unsqueeze(0).unsqueeze(-1)
    last_date = train_val_df['date'].max()
    future_preds = []

    with torch.no_grad():
        for _ in range(horizon):
            lstm_model.eval()
            pred_scaled = lstm_model(current_seq).item()
            pred = pred_scaled * std + mean
            next_date = last_date + pd.DateOffset(months=len(future_preds) + 1)
            future_preds.append({'date': next_date, 'predicted_ppi': pred})
            # 滚动窗口：lag 变预测值
            new_seq = np.append(last_seq[1:], pred_scaled).astype(np.float32)
            last_seq = new_seq
            current_seq = torch.from_numpy(last_seq).unsqueeze(0).unsqueeze(-1)

    return future_preds


# === 统一入口 ===
def train_all_monthly_models(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_lstm_params: Dict,
    target_col: str = 'ppi_index',
) -> Dict:
    """
    统一入口：训练 3 个模型（Prophet / XGBoost / LSTM）+ 评估 Final Test

    Args:
        train_val_df: Train + Validation (108 月)
        test_df: Final Test (24 月)
        best_lstm_params: P0.3 lstm_tuning.grid_search_lstm 输出的 best_params
        target_col: 目标列

    Returns:
        Dict 含 'prophet' / 'xgboost' / 'lstm' 三个 key，每个含 metrics + test_pred
    """
    _verify_data_boundary(train_val_df, test_df)

    result = {
        'status': 'success',
        'train_val_n': len(train_val_df),
        'test_n': len(test_df),
        'lstm_params': best_lstm_params,
    }

    # Prophet
    prophet_metrics, prophet_pred, prophet_model = train_prophet_final(train_val_df, test_df, target_col)
    result['prophet'] = {
        'metrics': prophet_metrics,
        'test_pred': prophet_pred,
        'test_actuals': test_df[target_col].values,
    }

    # XGBoost
    xgb_metrics, xgb_pred, xgb_model, xgb_history = train_xgboost_final(
        train_val_df, test_df, target_col
    )
    result['xgboost'] = {
        'metrics': xgb_metrics,
        'test_pred': xgb_pred,
        'test_actuals': test_df[target_col].values,
        'feature_cols': get_monthly_feature_columns(),
    }

    # LSTM (接 P0.3 best_params)
    lstm_metrics, lstm_pred, lstm_model, lstm_mean_std, lstm_actuals = train_lstm_final(
        train_val_df, test_df, best_lstm_params, target_col
    )
    result['lstm'] = {
        'metrics': lstm_metrics,
        'test_pred': lstm_pred,
        'test_actuals': lstm_actuals,
        'mean_std': lstm_mean_std,
    }

    return result


# === Synthetic LSTM Leakage Test ===
def test_synthetic_lstm_leakage(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: Dict,
    target_col: str = 'ppi_index',
) -> Dict:
    """
    Synthetic LSTM Leakage Test:
    修改 test 段某个未来 actual，验证所有 24 个 prediction 不受影响。

    验证逻辑：
    - 跑原始 test_df（正常版本），得到 24 个 predictions
    - 修改 test_df 最后一行（2025-12）的 ppi_index 为 999.0（极端值）
    - 再跑一次，得到 24 个 predictions
    - 比较两次 predictions：必须完全一致
       因为预测 2025-12 时 history_scaled[-6:] = 2025-06~2025-11 actual
       不含 2025-12 actual 自身 → 2025-12 的修改不影响任何预测
    """
    # 正常版本
    _, preds_normal, _, _, _ = train_lstm_final(
        train_val_df, test_df, best_params, target_col
    )

    # 修改版本
    test_modified = test_df.copy()
    test_modified.iloc[-1, test_modified.columns.get_loc(target_col)] = 999.0
    _, preds_modified, _, _, _ = train_lstm_final(
        train_val_df, test_modified, best_params, target_col
    )

    if not np.allclose(preds_normal, preds_modified, atol=1e-3):
        max_diff = float(np.max(np.abs(preds_normal - preds_modified)))
        return {
            'leakage_free': False,
            'details': (
                f"修改 test 最后一点 (2025-12) actual 为 999 后，"
                f"24 个 predictions 中存在不一致（最大差异 {max_diff:.6f}）。"
                f"说明 LSTM rolling 存在 future leakage。"
            ),
        }
    return {
        'leakage_free': True,
        'details': (
            f"修改 test 最后一点 (2025-12) actual 为 999 后，"
            f"24 个 predictions 完全一致（最大差异 < 1e-3）。"
            f"LSTM rolling 严格 causal，无 future leakage。"
        ),
    }


# === Synthetic Leakage Test ===
def test_synthetic_leakage(train_val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = 'ppi_index') -> Dict:
    """
    Synthetic Leakage Test:
    把 test 段某个未来点改成极端值，验证此前月份的特征不会受影响。

    关键检查：
    - 修改 test 段第 5 点的 ppi_index 为 999.0
    - 构造特征
    - 验证 test 段第 4 点（修改之前）的 lag/rolling/yoy/mom 特征不变
    - 验证 test 段第 6 点（修改之后）受极端值影响的特征被允许变化
       （因为历史用了真实值 5 点 = 999）

    Returns:
        {'leakage_free': bool, 'details': str}
    """
    full_df = pd.concat([train_val_df, test_df]).sort_values('date').reset_index(drop=True)

    # 原始特征
    features_orig = build_features_monthly_causal(full_df.copy(), target_col)

    # 修改 test 段第 5 点（index = train_val 长度 + 5）的 target 为 999
    test_start_idx = len(train_val_df)
    target_idx = test_start_idx + 5
    full_modified = full_df.copy()
    full_modified.iloc[target_idx, full_modified.columns.get_loc(target_col)] = 999.0
    features_modified = build_features_monthly_causal(full_modified, target_col)

    # 比较 test 段第 4 点（target_idx - 1）的特征
    check_idx = target_idx - 1
    rows_to_check = ['lag_1', 'lag_3', 'rolling_mean_3', 'yoy_change', 'mom_change']
    diffs = []
    for col in rows_to_check:
        if col in features_orig.columns and col in features_modified.columns:
            v_orig = features_orig.iloc[check_idx][col]
            v_mod = features_modified.iloc[check_idx][col]
            diff = abs(float(v_orig) - float(v_mod))
            diffs.append((col, diff))
            if diff > 1e-9:
                return {
                    'leakage_free': False,
                    'details': f"行 idx={check_idx}（修改点之前）特征 {col} 因未来点 {target_idx} 修改而变化（diff={diff:.6f}）",
                }

    return {
        'leakage_free': True,
        'details': f"测试点 idx={check_idx}（修改点之前）所有 {len(rows_to_check)} 个特征都未受未来点 {target_idx} 修改影响（diff<1e-9）",
        'diffs': diffs,
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ppi_monthly import load_monthly_ppi

    print('=' * 60)
    print('P0.4 monthly_lstm.py 单元测试')
    print('=' * 60)

    # === Test 1: 数据加载 + 边界切片 ===
    df_full = load_monthly_ppi()
    df_train_val = df_full[df_full['date'] <= '2023-12-31'].copy()
    df_test = df_full[df_full['date'] >= '2024-01-01'].copy()
    print(f'\n[Test 1] 数据切片:')
    print(f'  完整: {len(df_full)} 月')
    print(f'  Train+Val: {len(df_train_val)} 月')
    print(f'  Test: {len(df_test)} 月')
    print(f'  Train+Val 时间: {df_train_val["date"].min()} ~ {df_train_val["date"].max()}')
    print(f'  Test 时间: {df_test["date"].min()} ~ {df_test["date"].max()}')
    assert len(df_train_val) == 108, f"Train+Val 长度 {len(df_train_val)} != 108"
    assert len(df_test) == 24, f"Test 长度 {len(df_test)} != 24"

    # === Test 2: boundary assertion ===
    print(f'\n[Test 2] Boundary Assertion:')
    try:
        _verify_data_boundary(df_train_val, df_test)
        print('  ✓ 边界验证通过')
    except AssertionError as e:
        print(f'  ✗ 边界验证失败: {e}')

    # === Test 3: Synthetic Leakage Test ===
    print(f'\n[Test 3] Synthetic Leakage Test:')
    result = test_synthetic_leakage(df_train_val, df_test)
    if result['leakage_free']:
        print(f'  ✓ {result["details"]}')
    else:
        print(f'  ✗ {result["details"]}')

    # === Test 4: Prophet Final ===
    print(f'\n[Test 4] Prophet Final (out-of-sample):')
    prophet_metrics, prophet_pred, prophet_model = train_prophet_final(df_train_val, df_test)
    print(f'  Test metrics: {prophet_metrics}')

    # === Test 5: XGBoost Final ===
    print(f'\n[Test 5] XGBoost Final (rolling one-step-ahead):')
    xgb_metrics, xgb_pred, xgb_model, xgb_history = train_xgboost_final(df_train_val, df_test)
    print(f'  Test metrics: {xgb_metrics}')
    print(f'  Test pred shape: {xgb_pred.shape} (应为 24)')

    # === Test 6: LSTM Final (with P0.3 best_params) ===
    print(f'\n[Test 6] LSTM Final (接 P0.3 best_params):')
    best_params_p03 = {
        'hidden_size': 32, 'dropout': 0.1, 'seq_length': 6,
        'num_layers': 2, 'lr': 0.001,
    }
    print(f'  P0.3 best_params: {best_params_p03}')
    lstm_metrics, lstm_pred, lstm_model, lstm_mean_std, lstm_actuals = train_lstm_final(
        df_train_val, df_test, best_params_p03
    )
    print(f'  Test metrics: {lstm_metrics}')
    print(f'  Test pred shape: {lstm_pred.shape} (应为 24-seq_length=18)')

    # === Test 7: 统一入口 ===
    print(f'\n[Test 7] train_all_monthly_models 统一入口:')
    all_results = train_all_monthly_models(df_train_val, df_test, best_params_p03)
    print(f'  Status: {all_results["status"]}')
    for m in ['prophet', 'xgboost', 'lstm']:
        if m in all_results:
            print(f'  {m}: {all_results[m]["metrics"]}')

    print('\n' + '=' * 60)
    print('P0.4 monthly_lstm.py 测试完成')
    print('=' * 60)
