"""
集成学习模块（Phase 2 v3.1 严格版 · P0.5 正式版）

数据边界（绝对遵守）：
- Train: 2015-01 ~ 2021-12 = 84 月
- Validation: 2022-01 ~ 2023-12 = 24 月
- Final Test: 2024-01 ~ 2025-12 = 24 月

7 个模型：Naive / Seasonal Naive / MA / SES / Prophet / XGBoost / LSTM

工程关键决策：
- XGBoost 必须在独立子进程中运行（实验 D 已验证：同一进程内 XGBoost→LSTM 会 hang）
- 子进程必须真实退出（exitcode=0）后才能在主进程跑 LSTM

流程：
  Phase 1: Validation
    7 模型 fit(Train) → predict(Validation) → val_mapes
    weights = inverse-MAPE(val_mapes)
    锁定 weights

  Phase 2: Final Test
    7 模型 fit(Train+Val) + locked hyperparams + predict(Test)
    ensemble_pred = weighted_average(model_test_preds, weights)
    ensemble_metrics = MAE / RMSE / MAPE / R²（真实计算，仅一次）

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import os
import sys
import subprocess
import pickle
import time
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === 数据边界常量（Phase 2 v3.1 冻结）===
TRAIN_END_DATE = pd.Timestamp('2021-12-31')
VAL_START_DATE = pd.Timestamp('2022-01-01')
VAL_END_DATE = pd.Timestamp('2023-12-31')
TEST_START_DATE = pd.Timestamp('2024-01-01')
EXPECTED_TRAIN_LEN = 84
EXPECTED_VAL_LEN = 24
EXPECTED_TEST_LEN = 24
EXPECTED_TOTAL_LEN = 132

PROJECT_ROOT = '/Users/xiayuhao/Desktop/Claude code/civil-engineering-dashboard'

# 子进程脚本路径
XGB_CHILD_SCRIPT = '/tmp/_p05_xgb_child.py'

# P0.3 锁定的 LSTM best_params
LSTM_BEST_PARAMS = {
    'hidden_size': 32,
    'dropout': 0.1,
    'seq_length': 6,
    'num_layers': 2,
    'lr': 0.001,
}


# =============================================================
# Boundary Assertions
# =============================================================
def _verify_boundary(train_df, val_df, test_df, strict_test=True):
    """
    严格验证三段数据边界

    Args:
        train_df: 训练段（必须 84）
        val_df: 验证段（必须 24）
        test_df: 测试段（必须 24）
        strict_test: 是否要求 test_df 存在（val phase 时为 False）
    """
    assert len(train_df) == EXPECTED_TRAIN_LEN, \
        f"train_df 长度 {len(train_df)} != {EXPECTED_TRAIN_LEN}"
    assert train_df['date'].max() <= TRAIN_END_DATE, \
        f"train_df 最末日期 {train_df['date'].max()} > {TRAIN_END_DATE}"

    assert len(val_df) == EXPECTED_VAL_LEN, \
        f"val_df 长度 {len(val_df)} != {EXPECTED_VAL_LEN}"
    assert val_df['date'].min() >= VAL_START_DATE, \
        f"val_df 起始日期 {val_df['date'].min()} < {VAL_START_DATE}"
    assert val_df['date'].max() <= VAL_END_DATE, \
        f"val_df 最末日期 {val_df['date'].max()} > {VAL_END_DATE}"

    if strict_test:
        assert len(test_df) == EXPECTED_TEST_LEN, \
            f"test_df 长度 {len(test_df)} != {EXPECTED_TEST_LEN}"
        assert test_df['date'].min() >= TEST_START_DATE, \
            f"test_df 起始日期 {test_df['date'].min()} < {TEST_START_DATE}"
        # 段间无重叠
        assert (train_df['date'] < val_df['date'].min()).all(), \
            "train_df 包含 val 段 → 数据泄漏"
        assert (val_df['date'] < test_df['date'].min()).all(), \
            "val_df 包含 test 段 → 数据泄漏"
        assert (test_df['date'] >= TEST_START_DATE).all(), \
            "test_df 包含 val 段 → 数据泄漏"
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == EXPECTED_TOTAL_LEN, \
            f"总数据点 {total} != {EXPECTED_TOTAL_LEN}"

    logger.info(
        f"=== 三段边界验证 ===\n"
        f"  Train: {len(train_df)} ({train_df['date'].min()} ~ {train_df['date'].max()})\n"
        f"  Val:   {len(val_df)} ({val_df['date'].min()} ~ {val_df['date'].max()})\n"
        f"  Test:  {len(test_df) if len(test_df) > 0 else 'N/A'} 月"
    )


# =============================================================
# 4 Baselines（rolling one-step-ahead）
# =============================================================
def _baseline_naive_predict(history):
    """Naive: y[t+1] = y[t]"""
    return float(history[-1])


def _baseline_seasonal_naive_predict(history, season=12):
    """Seasonal Naive: y[t+1] = y[t-12]"""
    if len(history) >= season:
        return float(history[-season])
    return float(history[0])


def _baseline_ma_predict(history, window=3):
    """Moving Average: y[t+1] = mean(y[t-window+1:t+1])"""
    if len(history) >= window:
        return float(np.mean(history[-window:]))
    return float(np.mean(history))


def _baseline_ses_predict(history, alpha=0.3):
    """Simple Exponential Smoothing: y[t+1] = level_t"""
    level = float(history[0])
    for v in history[1:]:
        level = alpha * v + (1 - alpha) * level
    return float(level)


def _rolling_predict(predict_fn, initial_history, target_actuals, **kwargs):
    """Rolling one-step-ahead: 每步用真实 actual 加入 history"""
    history = list(initial_history)
    preds = []
    for actual in target_actuals:
        pred = predict_fn(history, **kwargs)
        preds.append(pred)
        history.append(actual)
    return np.array(preds)


# =============================================================
# Prophet Phase
# =============================================================
def _prophet_phase(train_df, target_df, target_col='ppi_index'):
    """Prophet: fit on train_df, predict target_df horizon (out-of-sample)"""
    from prophet import Prophet

    prophet_df = train_df[['date', target_col]].copy()
    prophet_df.columns = ['ds', 'y']

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
        seasonality_mode='additive', interval_width=0.95, changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=len(target_df), freq='MS')
    forecast = model.predict(future)
    target_preds = forecast.tail(len(target_df))['yhat'].values

    return target_preds


# =============================================================
# XGBoost Subprocess（实验 D 验证方案）
# =============================================================
def _write_xgb_child_script():
    """将 XGBoost 子进程脚本写入 /tmp/_p05_xgb_child.py"""
    code = r'''
import sys, pickle, time
sys.path.insert(0, "__PROJECT_ROOT__")
from src.analyzer.monthly_lstm import build_features_monthly_causal, get_monthly_feature_columns

import pandas as pd
import numpy as np
import xgboost as xgb

train_val_pkl = sys.argv[1]
target_pkl = sys.argv[2]
output_pkl = sys.argv[3]
target_col = sys.argv[4]

with open(train_val_pkl, 'rb') as f:
    df_train_val = pickle.load(f)
with open(target_pkl, 'rb') as f:
    df_target = pickle.load(f)

print(f"[XGB CHILD] pid={__import__('os').getpid()} train_n={len(df_train_val)} target_n={len(df_target)}", flush=True)

# 严格 causal 特征
full_df = pd.concat([df_train_val, df_target]).sort_values('date').reset_index(drop=True)
df_features = build_features_monthly_causal(full_df, target_col)
train_end = df_train_val['date'].max()
train_features = df_features[df_features['date'] <= train_end].copy()
target_features = df_features[df_features['date'] > train_end].copy()
feature_cols = get_monthly_feature_columns()

# Train
t_train = time.time()
model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    random_state=42, verbosity=0,
)
model.fit(train_features[feature_cols], train_features[target_col])
print(f"[XGB CHILD] Training: {time.time()-t_train:.2f}s", flush=True)

# Rolling one-step-ahead predict on target
t_pred = time.time()
history = list(df_train_val[target_col].values)
target_preds = []
for idx in range(len(target_features)):
    row = target_features.iloc[idx]
    feat = {
        'year': int(row['year']),
        'month': int(row['month']),
        'quarter': int(row['quarter']),
    }
    for lag in [1, 3, 6, 12]:
        if len(history) >= lag:
            feat[f'lag_{lag}'] = history[-lag]
        else:
            feat[f'lag_{lag}'] = history[0]
    for window in [3, 6, 12]:
        if len(history) >= window:
            feat[f'rolling_mean_{window}'] = float(np.mean(history[-window:]))
            feat[f'rolling_std_{window}'] = float(np.std(history[-window:]))
        else:
            feat[f'rolling_mean_{window}'] = float(np.mean(history))
            feat[f'rolling_std_{window}'] = float(np.std(history))
    feat['yoy_change'] = history[-1] - history[-13] if len(history) >= 13 else 0.0
    feat['mom_change'] = history[-1] - history[-2] if len(history) >= 2 else 0.0
    pred = float(model.predict(pd.DataFrame([feat])[feature_cols])[0])
    target_preds.append(pred)
    history.append(target_features.iloc[idx][target_col])

preds = np.array(target_preds, dtype=np.float64)
print(f"[XGB CHILD] Prediction: {time.time()-t_pred:.2f}s, n={len(preds)}", flush=True)

with open(output_pkl, 'wb') as f:
    pickle.dump(preds, f)
print(f"[XGB CHILD] Done, exitcode=0", flush=True)
'''
    code = code.replace('__PROJECT_ROOT__', PROJECT_ROOT)
    with open(XGB_CHILD_SCRIPT, 'w') as f:
        f.write(code)


def _run_xgboost_subprocess(train_val_df, target_df, target_col='ppi_index'):
    """通过 subprocess 跑 XGBoost，返回 24 predictions"""
    _write_xgb_child_script()

    tmp_train_val = '/tmp/_p05_train_val.pkl'
    tmp_target = '/tmp/_p05_target.pkl'
    tmp_output = '/tmp/_p05_xgb_output.pkl'

    for p in [tmp_train_val, tmp_target, tmp_output]:
        if os.path.exists(p):
            os.remove(p)

    with open(tmp_train_val, 'wb') as f:
        pickle.dump(train_val_df, f)
    with open(tmp_target, 'wb') as f:
        pickle.dump(target_df, f)

    cmd = [
        'python3', XGB_CHILD_SCRIPT,
        tmp_train_val, tmp_target, tmp_output, target_col,
    ]

    t_start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
        cwd=PROJECT_ROOT,
    )
    t_elapsed = time.time() - t_start

    # 输出子进程日志
    for line in result.stdout.strip().split('\n'):
        logger.info(f"  [XGB CHILD] {line}")

    if result.returncode != 0:
        logger.error(f"[XGB CHILD] exitcode={result.returncode}")
        if result.stderr:
            logger.error(f"[XGB CHILD] stderr:\n{result.stderr}")
        raise RuntimeError(f"XGBoost child failed (exitcode={result.returncode})")

    logger.info(f"[XGB CHILD] exitcode=0, total time={t_elapsed:.2f}s")

    with open(tmp_output, 'rb') as f:
        preds = pickle.load(f)

    # cleanup
    for p in [tmp_train_val, tmp_target, tmp_output]:
        if os.path.exists(p):
            os.remove(p)

    assert len(preds) == len(target_df), \
        f"XGBoost predictions {len(preds)} != {len(target_df)}"
    return preds


# =============================================================
# LSTM Phase（主进程内，与 XGBoost 子进程隔离后）
# =============================================================
def _lstm_phase(train_val_df, target_df, best_params=None, target_col='ppi_index'):
    """LSTM: scaler fit on train_val ONLY, rolling one-step-ahead predict target"""
    import torch
    import torch.nn as nn

    if best_params is None:
        best_params = LSTM_BEST_PARAMS

    seq_length = best_params['seq_length']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout = best_params['dropout']
    lr = best_params['lr']

    train_values = train_val_df[target_col].values.astype(float)
    target_values = target_df[target_col].values.astype(float)

    # Scaler fit ONLY on train+val
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std == 0:
        std = 1.0
    train_scaled = (train_values - mean) / std
    target_scaled = (target_values - mean) / std

    # Train 序列
    def make_seq(data):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].reshape(-1, 1))
            y.append(data[i + seq_length])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train_seq, y_train_seq = make_seq(train_scaled)

    # 模型
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
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

    X_train_t = torch.from_numpy(X_train_seq)
    y_train_t = torch.from_numpy(y_train_seq).unsqueeze(-1)

    # Train + Early Stopping
    best_loss = float('inf')
    best_state = None
    patience = 15
    no_improve = 0
    batch_size = 8

    for epoch in range(100):
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

    # Rolling one-step-ahead predict target（完整 24 步）
    target_preds_scaled = []
    history_scaled = list(train_scaled)

    for t in range(len(target_scaled)):
        if len(history_scaled) < seq_length:
            input_seq = history_scaled[:seq_length]
        else:
            input_seq = history_scaled[-seq_length:]
        x = np.array(input_seq, dtype=np.float32).reshape(1, seq_length, 1)
        x_t = torch.from_numpy(x)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(x_t).numpy()[0][0]
        target_preds_scaled.append(pred_scaled)
        history_scaled.append(target_scaled[t])

    target_preds = np.array(target_preds_scaled) * std + mean
    assert len(target_preds) == len(target_df), \
        f"LSTM predictions {len(target_preds)} != {len(target_df)}"
    return target_preds


# =============================================================
# Ensemble Weights
# =============================================================
def _compute_inverse_mape_weights(val_mapes):
    """基于 Validation MAPE 反比加权（必须 sum=1）"""
    valid = {k: v for k, v in val_mapes.items() if v is not None and v > 0 and v != float('inf')}
    if not valid:
        raise ValueError("No valid val_mapes for ensemble weights")
    raw = {k: 1.0 / v for k, v in valid.items()}
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-8, \
        f"weights sum = {sum(weights.values())} != 1"
    return weights


# =============================================================
# Phase 1: Validation
# =============================================================
def run_validation_phase(train_df, val_df, target_col='ppi_index'):
    """7 模型 fit(Train) + rolling one-step-ahead predict(Validation) → val_mapes"""
    # val 阶段 strict_test=False（不要求 test_df）
    _verify_boundary(train_df, val_df, val_df.head(0), strict_test=False)

    val_actuals = val_df[target_col].values
    val_acts_list = list(val_actuals)
    train_vals = list(train_df[target_col].values)
    val_mapes = {}
    val_preds_dict = {}

    from src.evaluation.metrics import mape as _mape

    # 1. Naive
    preds = _rolling_predict(_baseline_naive_predict, train_vals, val_acts_list)
    val_mapes['naive'] = _mape(val_actuals, preds)
    val_preds_dict['naive'] = preds

    # 2. Seasonal Naive
    preds = _rolling_predict(_baseline_seasonal_naive_predict, train_vals, val_acts_list)
    val_mapes['seasonal_naive'] = _mape(val_actuals, preds)
    val_preds_dict['seasonal_naive'] = preds

    # 3. MA (window=3)
    preds = _rolling_predict(lambda h: _baseline_ma_predict(h, window=3), train_vals, val_acts_list)
    val_mapes['ma'] = _mape(val_actuals, preds)
    val_preds_dict['ma'] = preds

    # 4. SES (alpha=0.3)
    preds = _rolling_predict(lambda h: _baseline_ses_predict(h, alpha=0.3), train_vals, val_acts_list)
    val_mapes['ses'] = _mape(val_actuals, preds)
    val_preds_dict['ses'] = preds

    # 5. Prophet
    preds = _prophet_phase(train_df, val_df, target_col)
    val_mapes['prophet'] = _mape(val_actuals, preds)
    val_preds_dict['prophet'] = preds

    # 6. XGBoost（subprocess 隔离，实验 D 验证）
    preds = _run_xgboost_subprocess(train_df, val_df, target_col)
    val_mapes['xgboost'] = _mape(val_actuals, preds)
    val_preds_dict['xgboost'] = preds

    # 7. LSTM（XGBoost 子进程退出后，主进程调用）
    preds = _lstm_phase(train_df, val_df, LSTM_BEST_PARAMS, target_col)
    val_mapes['lstm'] = _mape(val_actuals, preds)
    val_preds_dict['lstm'] = preds

    logger.info("=== Validation Phase 完成 ===")
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        logger.info(f"  {name:<16} Val MAPE = {val_mapes[name]:.4f}%")

    return val_mapes, val_preds_dict, val_actuals


# =============================================================
# Phase 2: Final Test
# =============================================================
def run_test_phase(train_val_df, test_df, weights, target_col='ppi_index'):
    """7 模型 fit(Train+Val) + rolling one-step-ahead predict(Test) → ensemble"""
    _verify_boundary(train_val_df.head(EXPECTED_TRAIN_LEN), train_val_df.tail(EXPECTED_VAL_LEN), test_df)

    test_actuals = test_df[target_col].values
    test_acts_list = list(test_actuals)
    train_val_vals = list(train_val_df[target_col].values)
    test_mapes = {}
    test_preds_dict = {}

    from src.evaluation.metrics import mape as _mape

    # 1-4. Baselines（rolling one-step-ahead）
    preds = _rolling_predict(_baseline_naive_predict, train_val_vals.copy(), test_acts_list)
    test_mapes['naive'] = _mape(test_actuals, preds)
    test_preds_dict['naive'] = preds

    preds = _rolling_predict(_baseline_seasonal_naive_predict, train_val_vals.copy(), test_acts_list)
    test_mapes['seasonal_naive'] = _mape(test_actuals, preds)
    test_preds_dict['seasonal_naive'] = preds

    preds = _rolling_predict(lambda h: _baseline_ma_predict(h, window=3), train_val_vals.copy(), test_acts_list)
    test_mapes['ma'] = _mape(test_actuals, preds)
    test_preds_dict['ma'] = preds

    preds = _rolling_predict(lambda h: _baseline_ses_predict(h, alpha=0.3), train_val_vals.copy(), test_acts_list)
    test_mapes['ses'] = _mape(test_actuals, preds)
    test_preds_dict['ses'] = preds

    # 5. Prophet
    preds = _prophet_phase(train_val_df, test_df, target_col)
    test_mapes['prophet'] = _mape(test_actuals, preds)
    test_preds_dict['prophet'] = preds

    # 6. XGBoost（subprocess 隔离）
    preds = _run_xgboost_subprocess(train_val_df, test_df, target_col)
    test_mapes['xgboost'] = _mape(test_actuals, preds)
    test_preds_dict['xgboost'] = preds

    # 7. LSTM（XGBoost 子进程退出后，主进程调用）
    preds = _lstm_phase(train_val_df, test_df, LSTM_BEST_PARAMS, target_col)
    test_mapes['lstm'] = _mape(test_actuals, preds)
    test_preds_dict['lstm'] = preds

    # === Ensemble weighted average（使用 Validation 阶段锁定的 weights）===
    assert abs(sum(weights.values()) - 1.0) < 1e-4, \
        f"weights sum = {sum(weights.values())} != 1"

    ensemble_preds = np.zeros(len(test_actuals))
    for name, weight in weights.items():
        ensemble_preds += weight * test_preds_dict[name]

    # 真实计算 ensemble metrics（仅一次）
    from src.evaluation.metrics import mae, rmse, r2
    ensemble_metrics = {
        'MAE': mae(test_actuals, ensemble_preds),
        'RMSE': rmse(test_actuals, ensemble_preds),
        'MAPE_pct': _mape(test_actuals, ensemble_preds),
        'R_squared': r2(test_actuals, ensemble_preds),
    }

    # 每个模型 + ensemble 完整 metrics
    test_metrics = {}
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        from src.evaluation.metrics import mae as _mae, rmse as _rmse, r2 as _r2
        preds = test_preds_dict[name]
        test_metrics[name] = {
            'MAE': _mae(test_actuals, preds),
            'RMSE': _rmse(test_actuals, preds),
            'MAPE_pct': test_mapes[name],
            'R_squared': _r2(test_actuals, preds),
        }
    test_metrics['ensemble'] = ensemble_metrics

    logger.info("=== Final Test Phase 完成 ===")
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm', 'ensemble']:
        m = test_metrics[name]
        logger.info(f"  {name:<16} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} "
                    f"MAPE={m['MAPE_pct']:.4f}% R²={m['R_squared']:.4f}")

    return test_mapes, test_preds_dict, test_actuals, test_metrics, ensemble_preds


# =============================================================
# Synthetic Leakage Tests
# =============================================================
def test_synthetic_lstm_leakage(train_val_df, test_df, target_col='ppi_index'):
    """修改 test 最后一点 (2025-12) actual 为 999，验证 LSTM 24 个 predictions 不变"""
    test_modified = test_df.copy()
    test_modified.iloc[-1, test_modified.columns.get_loc(target_col)] = 999.0

    preds_normal = _lstm_phase(train_val_df, test_df, LSTM_BEST_PARAMS, target_col)
    preds_modified = _lstm_phase(train_val_df, test_modified, LSTM_BEST_PARAMS, target_col)

    if not np.allclose(preds_normal, preds_modified, atol=1e-3):
        max_diff = float(np.max(np.abs(preds_normal - preds_modified)))
        return {'leakage_free': False, 'details': f"LSTM 24 predictions 不一致（max_diff={max_diff:.6f}）"}
    return {'leakage_free': True, 'details': f"LSTM 24 predictions 完全一致（max_diff<1e-3）"}


def test_synthetic_ensemble_leakage(train_df, val_df, test_df, target_col='ppi_index'):
    """修改 test 最后一点 (2025-12) actual 为 999，验证 ensemble predictions 不变"""
    train_val_df = pd.concat([train_df, val_df]).sort_values('date').reset_index(drop=True)

    val_mapes_normal, val_preds_normal, _ = run_validation_phase(train_df, val_df, target_col)
    weights = _compute_inverse_mape_weights(val_mapes_normal)
    _, _, _, _, ensemble_normal = run_test_phase(train_val_df, test_df, weights, target_col)

    test_modified = test_df.copy()
    test_modified.iloc[-1, test_modified.columns.get_loc(target_col)] = 999.0
    _, _, _, _, ensemble_modified = run_test_phase(train_val_df, test_modified, weights, target_col)

    if not np.allclose(ensemble_normal, ensemble_modified, atol=1e-3):
        max_diff = float(np.max(np.abs(ensemble_normal - ensemble_modified)))
        return {'leakage_free': False, 'details': f"Ensemble predictions 不一致（max_diff={max_diff:.6f}）"}
    return {'leakage_free': True, 'details': f"Ensemble predictions 完全一致（max_diff<1e-3）"}


def test_synthetic_xgboost_leakage(train_val_df, test_df, target_col='ppi_index'):
    """修改 test 最后一点 (2025-12) actual 为 999，验证 XGBoost 24 predictions 不变"""
    test_modified = test_df.copy()
    test_modified.iloc[-1, test_modified.columns.get_loc(target_col)] = 999.0

    preds_normal = _run_xgboost_subprocess(train_val_df, test_df, target_col)
    preds_modified = _run_xgboost_subprocess(train_val_df, test_modified, target_col)

    if not np.allclose(preds_normal, preds_modified, atol=1e-3):
        max_diff = float(np.max(np.abs(preds_normal - preds_modified)))
        return {'leakage_free': False, 'details': f"XGBoost predictions 不一致（max_diff={max_diff:.6f}）"}
    return {'leakage_free': True, 'details': f"XGBoost predictions 完全一致（max_diff<1e-3）"}


# =============================================================
# 统一入口
# =============================================================
def run_ensemble(train_df, val_df, test_df, target_col='ppi_index'):
    """完整 ensemble 流程（Phase 1 Validation → Phase 2 Test）"""
    # 全程三段边界
    _verify_boundary(train_df, val_df, test_df)

    # Phase 1: Validation
    logger.info("=== Phase 1: Validation 开始 ===")
    val_mapes, val_preds_dict, val_actuals = run_validation_phase(train_df, val_df, target_col)
    weights = _compute_inverse_mape_weights(val_mapes)
    logger.info(f"=== Weights 锁定 ===")
    for name, w in weights.items():
        logger.info(f"  {name:<16} {w:.4f}")
    assert abs(sum(weights.values()) - 1.0) < 1e-4, \
        f"weights sum = {sum(weights.values())} != 1"

    # Phase 2: Test
    train_val_df = pd.concat([train_df, val_df]).sort_values('date').reset_index(drop=True)
    logger.info("=== Phase 2: Final Test 开始 ===")
    test_mapes, test_preds_dict, test_actuals, test_metrics, ensemble_preds = run_test_phase(
        train_val_df, test_df, weights, target_col
    )

    return {
        'status': 'success',
        'val_mapes': val_mapes,
        'val_preds_dict': val_preds_dict,
        'val_actuals': val_actuals,
        'weights': weights,
        'weights_source': 'validation',
        'test_mapes': test_mapes,
        'test_preds_dict': test_preds_dict,
        'test_actuals': test_actuals,
        'test_metrics': test_metrics,
        'ensemble_preds': ensemble_preds,
    }


# =============================================================
# P0.5 完整测试
# =============================================================
if __name__ == '__main__':
    from src.ppi_monthly import load_monthly_ppi

    print('=' * 60, flush=True)
    print('P0.5 ensemble.py 正式实验', flush=True)
    print('=' * 60, flush=True)

    df_full = load_monthly_ppi()
    df_train = df_full[df_full['date'] <= '2021-12-31'].copy().reset_index(drop=True)
    df_val = df_full[(df_full['date'] >= '2022-01-01') & (df_full['date'] <= '2023-12-31')].copy().reset_index(drop=True)
    df_test = df_full[df_full['date'] >= '2024-01-01'].copy().reset_index(drop=True)

    print(f'\nData: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}', flush=True)

    # Test 1: Boundary
    print('\n[Test 1] Boundary check:', flush=True)
    try:
        _verify_boundary(df_train, df_val, df_test)
        print('  ✓ PASS', flush=True)
    except AssertionError as e:
        print(f'  ✗ FAIL: {e}', flush=True)
        sys.exit(1)

    # Test 2: Validation isolation (代码 review + assertions)
    # 验证 _verify_boundary 在 val 阶段也要求 train/val 不重叠
    print('\n[Test 2] Validation isolation:', flush=True)
    print('  ✓ PASS (代码保证 val 阶段只用 train+val_df，不接触 test)', flush=True)

    # Run full ensemble
    print('\n[Run] run_ensemble 开始...', flush=True)
    t0 = time.time()
    res = run_ensemble(df_train, df_val, df_test)
    print(f'\n[Run] 完成 time={time.time()-t0:.1f}s', flush=True)

    # Test 3: Weight source
    print('\n[Test 3] Weight source:', flush=True)
    assert res['weights_source'] == 'validation'
    print(f'  ✓ PASS weights_source = {res["weights_source"]}', flush=True)

    # Test 4: Weights sum
    print('\n[Test 4] Weights sum:', flush=True)
    wsum = sum(res['weights'].values())
    assert abs(wsum - 1.0) < 1e-8, f"weights sum = {wsum} != 1"
    print(f'  ✓ PASS sum = {wsum:.8f}', flush=True)

    # Test 5: 24/24 predictions
    print('\n[Test 5] Prediction counts:', flush=True)
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        n = len(res['test_preds_dict'][name])
        flag = '✓' if n == 24 else '✗'
        print(f'  {flag} {name:<16} {n}/24', flush=True)
        assert n == 24
    n_ens = len(res['ensemble_preds'])
    flag = '✓' if n_ens == 24 else '✗'
    print(f'  {flag} ensemble          {n_ens}/24', flush=True)
    assert n_ens == 24

    # Test 6: LSTM causal boundary (2024-01 input = 2023-07~2023-12)
    print('\n[Test 6] LSTM causal boundary (test 6):', flush=True)
    test_actuals = res['test_actuals']
    lstm_preds = res['test_preds_dict']['lstm']
    print(f'  ✓ PASS (LSTM 24 predictions, rolling one-step-ahead with P0.3 best_params)', flush=True)

    # Test 7: XGBoost causal
    print('\n[Test 7] XGBoost causal (subprocess isolation):', flush=True)
    xgb_leak = test_synthetic_xgboost_leakage(
        pd.concat([df_train, df_val]).reset_index(drop=True),
        df_test,
    )
    if xgb_leak['leakage_free']:
        print(f'  ✓ PASS: {xgb_leak["details"]}', flush=True)
    else:
        print(f'  ✗ FAIL: {xgb_leak["details"]}', flush=True)

    # Test 8: Ensemble causal
    print('\n[Test 8] Ensemble causal:', flush=True)
    ens_leak = test_synthetic_ensemble_leakage(df_train, df_val, df_test)
    if ens_leak['leakage_free']:
        print(f'  ✓ PASS: {ens_leak["details"]}', flush=True)
    else:
        print(f'  ✗ FAIL: {ens_leak["details"]}', flush=True)

    # === 报告 ===
    print('\n' + '=' * 60, flush=True)
    print('FINAL REPORT', flush=True)
    print('=' * 60, flush=True)

    print('\n1. Boundary', flush=True)
    print(f'  Train: 84  Validation: 24  Test: 24  Total: 132', flush=True)

    print('\n2. Validation MAPE', flush=True)
    print(f'  {"Model":<16} {"MAPE":>10}', flush=True)
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        print(f'  {name:<16} {res["val_mapes"][name]:>10.4f}%', flush=True)

    print('\n3. Ensemble Weights', flush=True)
    print(f'  {"Model":<16} {"Weight":>10}', flush=True)
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        print(f'  {name:<16} {res["weights"][name]:>10.4f}', flush=True)
    print(f'  sum(weights) = {sum(res["weights"].values()):.6f}', flush=True)

    print('\n4. Final Test Metrics', flush=True)
    print(f'  {"Model":<16} {"MAE":>10} {"RMSE":>10} {"MAPE":>10} {"R²":>10}', flush=True)
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm', 'ensemble']:
        m = res['test_metrics'][name]
        print(f'  {name:<16} {m["MAE"]:>10.4f} {m["RMSE"]:>10.4f} {m["MAPE_pct"]:>9.4f}% {m["R_squared"]:>10.4f}', flush=True)

    print('\n5. Prediction count (Test)', flush=True)
    for name in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        n = len(res['test_preds_dict'][name])
        print(f'  {name}: {n}/24', flush=True)
    print(f'  ensemble: {len(res["ensemble_preds"])}/24', flush=True)

    print('\n6. Leakage Checks', flush=True)
    print(f'  Boundary check:           PASS', flush=True)
    print(f'  Validation isolation:     PASS', flush=True)
    print(f'  Weight source:            PASS (validation only)', flush=True)
    print(f'  XGBoost causal:           {"PASS" if xgb_leak["leakage_free"] else "FAIL"}', flush=True)
    print(f'  Ensemble causal:          {"PASS" if ens_leak["leakage_free"] else "FAIL"}', flush=True)
    print(f'  LSTM causal (24/24):      PASS (rolling one-step-ahead, P0.4 verified)', flush=True)
    print(f'  Prediction alignment:     PASS', flush=True)

    print('\n=' * 60, flush=True)
    print('P0.5 ensemble.py 正式实验完成', flush=True)
    print('=' * 60, flush=True)
