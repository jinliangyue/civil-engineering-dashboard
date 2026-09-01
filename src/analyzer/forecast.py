"""
预测模型模块
功能：基于历史数据预测未来 N 个月的价格
使用 Prophet（Facebook 开源）+ ARIMA 备选

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备 Prophet 输入数据
    Prophet 要求两列：ds（日期） + y（数值）
    """
    prophet_df = df[['date', 'price']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df


def train_prophet(df: pd.DataFrame, periods: int = 6, freq: str = 'MS') -> Optional[object]:
    """
    训练 Prophet 模型并预测未来 periods 个月
    返回 Prophet 模型对象 + 预测结果 DataFrame
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.error('prophet 未安装，请运行 pip install prophet')
        return None
    if df.empty or len(df) < 12:
        logger.warning(f'数据不足 12 个月（当前 {len(df)}），跳过 Prophet')
        return None
    prophet_df = prepare_prophet_data(df)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        interval_width=0.95,
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return {'model': model, 'forecast': forecast}


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict:
    """
    计算模型评估指标
    - MAE（平均绝对误差）
    - RMSE（均方根误差）
    - MAPE（平均绝对百分比误差）
    - R²（决定系数）
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    # 移除 NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        return {}
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if actual.mean() > 0 else 0
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'MAPE_pct': round(float(mape), 2),
        'R_squared': round(float(r_squared), 4),
    }


def train_arima(df: pd.DataFrame, periods: int = 6, order: Tuple = (1, 1, 1)) -> Optional[Dict]:
    """
    训练 ARIMA 模型作为备选方案
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        logger.error('statsmodels 未安装')
        return None
    if df.empty or len(df) < 24:
        logger.warning(f'ARIMA 需要至少 24 个月数据（当前 {len(df)}）')
        return None
    ts = df.set_index('date')['price'].sort_index().asfreq('MS').interpolate()
    try:
        model = ARIMA(ts, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=periods)
        # 在训练集上的预测（评估模型在训练数据上的拟合度）
        fitted_values = fitted.fittedvalues
        metrics = calculate_metrics(pd.Series(ts.values), pd.Series(fitted_values))
        return {
            'model': fitted,
            'forecast': forecast,
            'metrics': metrics,
        }
    except Exception as e:
        logger.error(f'ARIMA 训练失败: {e}')
        return None


def forecast_single_series(df: pd.DataFrame, periods: int = 6) -> Dict:
    """
    对单个材料-地区组合做预测
    返回 Prophet + ARIMA 两个模型的预测结果 + 评估指标
    """
    result = {'periods': periods}
    # Prophet
    prophet_result = train_prophet(df, periods=periods)
    if prophet_result:
        forecast_df = prophet_result['forecast']
        # 历史拟合（用于评估）
        historical = forecast_df[forecast_df['ds'] <= df['date'].max()]
        y_true = df.set_index('date')['price'].reindex(historical['ds']).values
        y_pred = historical['yhat'].values
        metrics = calculate_metrics(pd.Series(y_true), pd.Series(y_pred))
        result['prophet'] = {
            'forecast': forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods),
            'metrics': metrics,
        }
        result['prophet']['model'] = prophet_result['model']
    # ARIMA
    arima_result = train_arima(df, periods=periods)
    if arima_result:
        result['arima'] = {
            'forecast': arima_result['forecast'],
            'metrics': arima_result['metrics'],
        }
        result['arima']['model'] = arima_result['model']
    return result


def forecast_all(df: pd.DataFrame, periods: int = 6) -> Dict:
    """
    对所有材料-地区组合做预测
    """
    if df.empty:
        return {'status': 'empty'}
    result = {'status': 'success', 'forecasts': {}, 'summary': []}
    for (material, region), group in df.groupby(['material', 'region']):
        key = f'{material}_{region}'
        logger.info(f'预测: {key} ({len(group)} 个月数据)')
        forecast = forecast_single_series(group, periods=periods)
        result['forecasts'][key] = forecast
        # 汇总
        if 'prophet' in forecast and 'metrics' in forecast['prophet']:
            metrics = forecast['prophet']['metrics']
            summary_row = {
                'material': material,
                'region': region,
                'data_points': len(group),
                'prophet_mape': metrics.get('MAPE_pct', None),
                'prophet_r2': metrics.get('R_squared', None),
            }
            if 'arima' in forecast and 'metrics' in forecast['arima']:
                summary_row['arima_mape'] = forecast['arima']['metrics'].get('MAPE_pct', None)
            result['summary'].append(summary_row)
    result['summary_df'] = pd.DataFrame(result['summary'])
    return result


def format_forecast_output(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    格式化预测输出，便于展示
    """
    output = forecast_df.copy()
    output.columns = ['date', 'predicted_price', 'lower_bound_95', 'upper_bound_95']
    output['date'] = pd.to_datetime(output['date']).dt.strftime('%Y-%m')
    output['predicted_price'] = output['predicted_price'].round(2)
    output['lower_bound_95'] = output['lower_bound_95'].round(2)
    output['upper_bound_95'] = output['upper_bound_95'].round(2)
    return output


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from data_loader import load_all_raw
    from data_cleaner import clean_pipeline
    df_raw = load_all_raw()
    if df_raw.empty:
        print('没有数据')
    else:
        df_clean, _ = clean_pipeline(df_raw)
        result = forecast_all(df_clean, periods=6)
        print('\n=== 预测结果汇总 ===')
        if 'summary_df' in result:
            print(result['summary_df'])
        print('\n=== 单个预测示例 ===')
        if result['forecasts']:
            first_key = list(result['forecasts'].keys())[0]
            first_forecast = result['forecasts'][first_key]
            if 'prophet' in first_forecast:
                print(f'预测 {first_key}:')
                print(format_forecast_output(first_forecast['prophet']['forecast']))