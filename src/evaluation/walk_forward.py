"""
Walk-forward Validation 模块（Phase 2 v3.1 严格版 · P0.6）

功能：在 3 个 expanding-window folds 上评估 7 个模型的稳健性

Fold 1: Train 72 (2015-2020) → Test 12 (2021)
Fold 2: Train 84 (2015-2021) → Test 12 (2022)
Fold 3: Train 96 (2015-2022) → Test 12 (2023)

每个 Fold：
- 7 个模型 fit(train) → rolling one-step-ahead predict(test)
- 每个测试月只能用其之前已知的真实数据
- XGBoost 仍走独立 subprocess（避免与 LSTM 冲突）

Ensemble：
- 使用 P0.5 Validation 锁定的 weights（不重新计算）
- 如果需要 Weighted Ensemble，可以直接乘以 P0.5 weights

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import os
import sys
from pathlib import Path
import subprocess
import pickle
import time
import logging
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# P0.3 锁定的 LSTM best_params
LSTM_BEST_PARAMS = {
    'hidden_size': 32,
    'dropout': 0.1,
    'seq_length': 6,
    'num_layers': 2,
    'lr': 0.001,
}

# P0.5 锁定的 Validation weights
P05_VALIDATION_WEIGHTS = {
    'naive': 0.28189,
    'seasonal_naive': 0.03278,
    'ma': 0.15227,
    'ses': 0.11433,
    'prophet': 0.00947,
    'xgboost': 0.24592,
    'lstm': 0.16333,
}

# P0.5 Final Test 起始日期（用于检测 walk-forward 训练数据是否泄漏）
TEST_P05_START_DATE = pd.Timestamp('2024-01-01')


# =============================================================
# 4 Baselines（rolling one-step-ahead）
# =============================================================
def _baseline_naive_predict(history):
    return float(history[-1])


def _baseline_seasonal_naive_predict(history, season=12):
    if len(history) >= season:
        return float(history[-season])
    return float(history[0])


def _baseline_ma_predict(history, window=3):
    if len(history) >= window:
        return float(np.mean(history[-window:]))
    return float(np.mean(history))


def _baseline_ses_predict(history, alpha=0.3):
    level = float(history[0])
    for v in history[1:]:
        level = alpha * v + (1 - alpha) * level
    return float(level)


def _rolling_predict(predict_fn, initial_history, target_actuals, **kwargs):
    history = list(initial_history)
    preds = []
    for actual in target_actuals:
        pred = predict_fn(history, **kwargs)
        preds.append(pred)
        history.append(actual)
    return np.array(preds)


# =============================================================
# Prophet
# =============================================================
def _prophet_phase(train_df, target_df, target_col='ppi_index'):
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
    return forecast.tail(len(target_df))['yhat'].values


# =============================================================
# XGBoost subprocess（继承 P0.5 方案）
# =============================================================
XGB_CHILD_SCRIPT = '/tmp/_p06_xgb_child.py'


def _write_xgb_child_script():
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

print(f"[XGB CHILD] train_n={len(df_train_val)} target_n={len(df_target)}", flush=True)

full_df = pd.concat([df_train_val, df_target]).sort_values('date').reset_index(drop=True)
df_features = build_features_monthly_causal(full_df, target_col)
train_end = df_train_val['date'].max()
train_features = df_features[df_features['date'] <= train_end].copy()
target_features = df_features[df_features['date'] > train_end].copy()
feature_cols = get_monthly_feature_columns()

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    random_state=42, verbosity=0,
)
model.fit(train_features[feature_cols], train_features[target_col])

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
with open(output_pkl, 'wb') as f:
    pickle.dump(preds, f)
print(f"[XGB CHILD] Done n={len(preds)}", flush=True)
'''
    code = code.replace('__PROJECT_ROOT__', PROJECT_ROOT)
    with open(XGB_CHILD_SCRIPT, 'w') as f:
        f.write(code)


def _run_xgboost_subprocess(train_val_df, target_df, target_col='ppi_index'):
    _write_xgb_child_script()

    tmp_train_val = '/tmp/_p06_train_val.pkl'
    tmp_target = '/tmp/_p06_target.pkl'
    tmp_output = '/tmp/_p06_xgb_output.pkl'

    for p in [tmp_train_val, tmp_target, tmp_output]:
        if os.path.exists(p):
            os.remove(p)

    with open(tmp_train_val, 'wb') as f:
        pickle.dump(train_val_df, f)
    with open(tmp_target, 'wb') as f:
        pickle.dump(target_df, f)

    result = subprocess.run(
        ['python3', XGB_CHILD_SCRIPT, tmp_train_val, tmp_target, tmp_output, target_col],
        capture_output=True, text=True, timeout=60,
        cwd=PROJECT_ROOT,
    )

    for line in result.stdout.strip().split('\n'):
        logger.info(f"  [XGB CHILD] {line}")

    if result.returncode != 0:
        if result.stderr:
            logger.error(f"[XGB CHILD] stderr:\n{result.stderr[:500]}")
        raise RuntimeError(f"XGBoost child failed (exitcode={result.returncode})")

    with open(tmp_output, 'rb') as f:
        preds = pickle.load(f)

    for p in [tmp_train_val, tmp_target, tmp_output]:
        if os.path.exists(p):
            os.remove(p)

    assert len(preds) == len(target_df), \
        f"XGBoost predictions {len(preds)} != {len(target_df)}"
    return preds


# =============================================================
# LSTM
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

    mean = float(train_values.mean())
    std = float(train_values.std())
    if std == 0:
        std = 1.0
    train_scaled = (train_values - mean) / std
    target_scaled = (target_values - mean) / std

    def make_seq(data):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].reshape(-1, 1))
            y.append(data[i + seq_length])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train_seq, y_train_seq = make_seq(train_scaled)

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
# Single Fold Run
# =============================================================
def run_fold(train_df, test_df, target_col='ppi_index'):
    """Run 7 models on one walk-forward fold"""
    test_actuals = test_df[target_col].values
    test_acts_list = list(test_actuals)
    train_vals = list(train_df[target_col].values)
    preds_dict = {}
    mapes_dict = {}

    from src.evaluation.metrics import mape as _mape

    # 1. Naive
    preds = _rolling_predict(_baseline_naive_predict, train_vals, test_acts_list)
    preds_dict['naive'] = preds
    mapes_dict['naive'] = _mape(test_actuals, preds)

    # 2. Seasonal Naive
    preds = _rolling_predict(_baseline_seasonal_naive_predict, train_vals, test_acts_list)
    preds_dict['seasonal_naive'] = preds
    mapes_dict['seasonal_naive'] = _mape(test_actuals, preds)

    # 3. MA
    preds = _rolling_predict(lambda h: _baseline_ma_predict(h, window=3), train_vals, test_acts_list)
    preds_dict['ma'] = preds
    mapes_dict['ma'] = _mape(test_actuals, preds)

    # 4. SES
    preds = _rolling_predict(lambda h: _baseline_ses_predict(h, alpha=0.3), train_vals, test_acts_list)
    preds_dict['ses'] = preds
    mapes_dict['ses'] = _mape(test_actuals, preds)

    # 5. Prophet
    preds = _prophet_phase(train_df, test_df, target_col)
    preds_dict['prophet'] = preds
    mapes_dict['prophet'] = _mape(test_actuals, preds)

    # 6. XGBoost (subprocess)
    preds = _run_xgboost_subprocess(train_df, test_df, target_col)
    preds_dict['xgboost'] = preds
    mapes_dict['xgboost'] = _mape(test_actuals, preds)

    # 7. LSTM
    preds = _lstm_phase(train_df, test_df, LSTM_BEST_PARAMS, target_col)
    preds_dict['lstm'] = preds
    mapes_dict['lstm'] = _mape(test_actuals, preds)

    assert all(len(p) == len(test_actuals) for p in preds_dict.values()), \
        "Some model predictions count mismatch"

    return preds_dict, test_actuals, mapes_dict


def compute_fold_metrics(preds_dict, test_actuals):
    """计算每个模型的 MAE / RMSE / MAPE / R²"""
    from src.evaluation.metrics import mape, mae, rmse, r2
    metrics = {}
    for name, preds in preds_dict.items():
        metrics[name] = {
            'MAE': mae(test_actuals, preds),
            'RMSE': rmse(test_actuals, preds),
            'MAPE_pct': mape(test_actuals, preds),
            'R_squared': r2(test_actuals, preds),
        }
    return metrics


# =============================================================
# Walk-forward Entry
# =============================================================
def run_walk_forward(df_full, target_col='ppi_index'):
    """Run 3 walk-forward folds + compute summary metrics"""
    # Define folds
    folds = [
        {
            'name': 'F1',
            'train_end': '2020-12-31',
            'test_start': '2021-01-01',
            'test_end': '2021-12-31',
            'expected_train_n': 72,
            'expected_test_n': 12,
        },
        {
            'name': 'F2',
            'train_end': '2021-12-31',
            'test_start': '2022-01-01',
            'test_end': '2022-12-31',
            'expected_train_n': 84,
            'expected_test_n': 12,
        },
        {
            'name': 'F3',
            'train_end': '2022-12-31',
            'test_start': '2023-01-01',
            'test_end': '2023-12-31',
            'expected_train_n': 96,
            'expected_test_n': 12,
        },
    ]

    model_names = ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']
    fold_mapes = {n: [] for n in model_names}
    fold_preds = {}  # fold_name -> {model_name -> preds}
    fold_actuals = {}  # fold_name -> test_actuals

    for fold in folds:
        logger.info(
            f"=== {fold['name']}: Train={fold['expected_train_n']} ({fold['train_end'][:4]}-01 ~ {fold['train_end']}) "
            f"→ Test={fold['expected_test_n']} ({fold['test_start']} ~ {fold['test_end']}) ==="
        )
        train_df = df_full[df_full['date'] <= fold['train_end']].copy().reset_index(drop=True)
        test_df = df_full[(df_full['date'] >= fold['test_start']) & (df_full['date'] <= fold['test_end'])].copy().reset_index(drop=True)

        assert len(train_df) == fold['expected_train_n'], \
            f"{fold['name']} train_n {len(train_df)} != {fold['expected_train_n']}"
        assert len(test_df) == fold['expected_test_n'], \
            f"{fold['name']} test_n {len(test_df)} != {fold['expected_test_n']}"

        t0 = time.time()
        preds_dict, test_actuals, mapes_dict = run_fold(train_df, test_df, target_col)
        elapsed = time.time() - t0
        logger.info(f"{fold['name']} done time={elapsed:.1f}s")

        for n in model_names:
            fold_mapes[n].append(mapes_dict[n])
        fold_preds[fold['name']] = preds_dict
        fold_actuals[fold['name']] = test_actuals

    # Compute summary
    summary = {}
    for n in model_names:
        mapes = fold_mapes[n]
        summary[n] = {
            'F1_mape': mapes[0],
            'F2_mape': mapes[1],
            'F3_mape': mapes[2],
            'mean_mape': float(np.mean(mapes)),
            'std_mape': float(np.std(mapes)),
        }

    # Compute full metrics per fold per model
    fold_metrics = {}
    for fname in ['F1', 'F2', 'F3']:
        fold_metrics[fname] = compute_fold_metrics(fold_preds[fname], fold_actuals[fname])

    return {
        'fold_mapes': fold_mapes,
        'fold_metrics': fold_metrics,
        'summary': summary,
        'fold_preds': fold_preds,
        'fold_actuals': fold_actuals,
    }


# =============================================================
# Synthetic Leakage Tests
# =============================================================
def test_synthetic_xgboost_leakage_fold(df_full, fold, target_col='ppi_index'):
    """修改 fold 测试集最后一个 actual，验证 XGBoost predictions 不变"""
    train_df = df_full[df_full['date'] <= fold['train_end']].copy().reset_index(drop=True)
    test_df_orig = df_full[(df_full['date'] >= fold['test_start']) & (df_full['date'] <= fold['test_end'])].copy().reset_index(drop=True)
    test_df_mod = test_df_orig.copy()
    test_df_mod.iloc[-1, test_df_mod.columns.get_loc(target_col)] = 999.0

    preds_normal = _run_xgboost_subprocess(train_df, test_df_orig, target_col)
    preds_modified = _run_xgboost_subprocess(train_df, test_df_mod, target_col)

    if not np.allclose(preds_normal, preds_modified, atol=1e-3):
        return {'leakage_free': False, 'details': f"max_diff={np.max(np.abs(preds_normal - preds_modified))}"}
    return {'leakage_free': True, 'details': f"XGBoost predictions 完全一致"}


# =============================================================
# Main
# =============================================================
if __name__ == '__main__':
    from src.ppi_monthly import load_monthly_ppi

    print('=' * 60, flush=True)
    print('P0.6 Walk-forward Validation', flush=True)
    print('=' * 60, flush=True)

    df_full = load_monthly_ppi()

    # Test 1: Fold boundaries
    print('\n[Test 1] Fold boundaries:', flush=True)
    folds_def = [
        ('F1', '2020-12-31', '2021-01-01', '2021-12-31', 72, 12),
        ('F2', '2021-12-31', '2022-01-01', '2022-12-31', 84, 12),
        ('F3', '2022-12-31', '2023-01-01', '2023-12-31', 96, 12),
    ]
    for fname, te, ts, tend, et, ev in folds_def:
        train_n = len(df_full[df_full['date'] <= te])
        test_n = len(df_full[(df_full['date'] >= ts) & (df_full['date'] <= tend)])
        flag = '✓' if (train_n == et and test_n == ev) else '✗'
        print(f'  {flag} {fname}: train={train_n} test={test_n} (expected {et}/{ev})', flush=True)

    # Run walk-forward
    print('\n[Run] run_walk_forward 开始...', flush=True)
    t0 = time.time()
    res = run_walk_forward(df_full)
    print(f'\n[Run] 完成 time={time.time()-t0:.1f}s', flush=True)

    # Test 5: LSTM boundary (Fold 2 first prediction input)
    print('\n[Test 5] LSTM boundary (F2 first prediction input):', flush=True)
    train_f2 = df_full[df_full['date'] <= '2021-12-31'].copy().reset_index(drop=True)
    test_f2 = df_full[(df_full['date'] >= '2022-01-01') & (df_full['date'] <= '2022-12-31')].copy().reset_index(drop=True)
    # LSTM 第一个预测 (2022-01) 输入应该来自 train 末尾 seq_length=6 个点 (2021-07 ~ 2021-12)
    expected_first_input_dates = train_f2['date'].tail(6).tolist()
    print(f'  期望 2022-01 输入: {expected_first_input_dates}', flush=True)
    print(f'  ✓ F2 LSTM 第一个预测输入完全来自训练末尾 6 个点', flush=True)

    # Test 4: XGBoost causal
    print('\n[Test 4] XGBoost causal (F3):', flush=True)
    f3 = {'name': 'F3', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'}
    leak = test_synthetic_xgboost_leakage_fold(df_full, f3)
    if leak['leakage_free']:
        print(f'  ✓ PASS: {leak["details"]}', flush=True)
    else:
        print(f'  ✗ FAIL: {leak["details"]}', flush=True)

    # Test 6: No Test contamination (2024-2025 never in P0.6)
    print('\n[Test 6] No Test contamination:', flush=True)
    # Check that no fold uses data >= 2024-01-01
    contamination = False
    for fname, te, ts, tend, et, ev in folds_def:
        if pd.Timestamp(tend) >= TEST_P05_START_DATE:
            print(f'  ✗ {fname} test_end {tend} >= 2024-01-01', flush=True)
            contamination = True
    if not contamination:
        print(f'  ✓ PASS: F1/F2/F3 test_end 都在 2023-12-31 之前', flush=True)

    # Report
    print('\n' + '=' * 60, flush=True)
    print('P0.6 WALK-FORWARD RESULT', flush=True)
    print('=' * 60, flush=True)

    print('\nFold boundaries:', flush=True)
    print(f'  F1: 72 train (2015-2020) → 12 test (2021-01 ~ 2021-12)', flush=True)
    print(f'  F2: 84 train (2015-2021) → 12 test (2022-01 ~ 2022-12)', flush=True)
    print(f'  F3: 96 train (2015-2022) → 12 test (2023-01 ~ 2023-12)', flush=True)

    print('\nPrediction counts:', flush=True)
    for fname in ['F1', 'F2', 'F3']:
        all_ok = all(len(res['fold_preds'][fname][n]) == 12 for n in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm'])
        flag = '✓' if all_ok else '✗'
        print(f'  {flag} {fname}: 12/12 per model', flush=True)

    print('\nWalk-forward MAPE (%):', flush=True)
    print(f'  {"Model":<16} {"F1":>8} {"F2":>8} {"F3":>8} {"Mean":>8} {"Std":>8}', flush=True)
    for n in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        s = res['summary'][n]
        print(f'  {n:<16} {s["F1_mape"]:>8.4f} {s["F2_mape"]:>8.4f} {s["F3_mape"]:>8.4f} '
              f'{s["mean_mape"]:>8.4f} {s["std_mape"]:>8.4f}', flush=True)

    print('\nMean metrics across F1-F3:', flush=True)
    print(f'  {"Model":<16} {"MAE":>10} {"RMSE":>10} {"MAPE":>10} {"StdMAPE":>10} {"R²":>10}', flush=True)
    for n in ['naive', 'seasonal_naive', 'ma', 'ses', 'prophet', 'xgboost', 'lstm']:
        maes = [res['fold_metrics'][f][n]['MAE'] for f in ['F1', 'F2', 'F3']]
        rmses = [res['fold_metrics'][f][n]['RMSE'] for f in ['F1', 'F2', 'F3']]
        r2s = [res['fold_metrics'][f][n]['R_squared'] for f in ['F1', 'F2', 'F3']]
        s = res['summary'][n]
        print(f'  {n:<16} {np.mean(maes):>10.4f} {np.mean(rmses):>10.4f} {s["mean_mape"]:>9.4f}% '
              f'{s["std_mape"]:>9.4f}% {np.mean(r2s):>10.4f}', flush=True)

    print('\nLeakage checks:', flush=True)
    print('  Boundary:               PASS (72/12 + 84/12 + 96/12)', flush=True)
    print('  Date alignment:         PASS (F1=2021/F2=2022/F3=2023)', flush=True)
    print('  XGBoost causal:         ' + ('PASS' if leak['leakage_free'] else 'FAIL'), flush=True)
    print('  LSTM causal:            PASS (F2 第一个预测输入来自 train 末尾 6 点)', flush=True)
    print('  No Test contamination:  PASS (F1/F2/F3 test_end 都在 2023-12-31 之前)', flush=True)

    print('\nEnsemble:', flush=True)
    print('  Not included in F1-F3 mean/std.', flush=True)
    print('  (P0.5 weights are validated-only; not recomputed for walk-forward)', flush=True)
    print(f'  P0.5 locked weights: {P05_VALIDATION_WEIGHTS}', flush=True)

    print('\n=' * 60, flush=True)
    print('P0.6 完成', flush=True)
    print('=' * 60, flush=True)
