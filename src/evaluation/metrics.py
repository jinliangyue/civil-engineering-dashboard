"""
项目评估指标统一模块

集中定义 4 个评估指标：
- mape(y_true, y_pred) -> float (百分比)
- mae(y_true, y_pred) -> float
- rmse(y_true, y_pred) -> float
- r2(y_true, y_pred) -> float

设计原则：
1. 输入支持 list / numpy.ndarray / pandas.Series
2. 不静默产生 inf / nan
3. 边界情况（除零、常量真实值、空输入）抛 ValueError
4. R² 与 sklearn.metrics.r2_score 语义一致

边界行为：
- MAPE: y_true=0 的样本不参与计算；若全部 y_true=0 → raise ValueError
- MAE / RMSE: 空输入 → raise ValueError
- R²: 若 y_true 全相等（方差为零）→ raise ValueError（与 sklearn 一致）
        若 y_true 与 y_pred 长度不一致 → raise ValueError

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

from typing import Union
import numpy as np
import pandas as pd

# 支持的输入类型
ArrayLike = Union[list, np.ndarray, pd.Series]


def _to_numpy(y: ArrayLike, name: str = "input") -> np.ndarray:
    """
    将输入统一转为 numpy 数组（float64）

    支持：list / tuple / numpy.ndarray / pandas.Series

    Raises:
        TypeError: 输入类型不支持
        ValueError: 输入为空
    """
    if isinstance(y, (list, tuple)):
        if len(y) == 0:
            raise ValueError(f"{name} is empty")
        arr = np.asarray(y, dtype=float)
    elif isinstance(y, np.ndarray):
        if y.size == 0:
            raise ValueError(f"{name} is empty")
        arr = y.astype(float)
    elif isinstance(y, pd.Series):
        if y.empty:
            raise ValueError(f"{name} is empty")
        arr = y.to_numpy(dtype=float)
    else:
        raise TypeError(
            f"{name} must be list, numpy.ndarray, or pandas.Series; "
            f"got {type(y).__name__}"
        )
    return arr


def _check_pair(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """检查两个数组长度一致且非空"""
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_pred) == 0:
        raise ValueError("y_pred is empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred length mismatch: "
            f"{len(y_true)} vs {len(y_pred)}"
        )


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Percentage Error (百分比)

    公式: mean(|y_true - y_pred| / |y_true|) * 100，仅在 y_true != 0 处计算

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        MAPE 百分比

    Raises:
        ValueError: 输入为空 / 长度不匹配 / 全部 y_true=0
        TypeError: 输入类型不支持

    Note:
        y_true=0 的样本会被过滤掉，不参与 MAPE 计算。
        如果所有 y_true=0（无法形成百分比误差），抛 ValueError。
    """
    yt = _to_numpy(y_true, "y_true")
    yp = _to_numpy(y_pred, "y_pred")
    _check_pair(yt, yp)

    mask = yt != 0
    if not mask.any():
        raise ValueError(
            "MAPE undefined: all y_true values are zero. "
            "Cannot compute percentage error against zero baseline."
        )

    yt_f = yt[mask]
    yp_f = yp[mask]
    return float(np.mean(np.abs((yt_f - yp_f) / yt_f)) * 100)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error

    公式: mean(|y_true - y_pred|)
    """
    yt = _to_numpy(y_true, "y_true")
    yp = _to_numpy(y_pred, "y_pred")
    _check_pair(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error

    公式: sqrt(mean((y_true - y_pred)^2))
    """
    yt = _to_numpy(y_true, "y_true")
    yp = _to_numpy(y_pred, "y_pred")
    _check_pair(yt, yp)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of Determination (R²)

    公式: 1 - SS_res / SS_tot
    其中 SS_res = sum((y_true - y_pred)^2)
         SS_tot = sum((y_true - mean(y_true))^2)

    语义与 sklearn.metrics.r2_score 一致：
    - sklearn 0.23+ 在 y_true 为常数时抛 ValueError
    - 本实现也抛 ValueError，明确说明「zero variance / constant target」

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        R² 值

    Raises:
        ValueError: 输入为空 / 长度不匹配 / y_true 为常数（方差为零）
        TypeError: 输入类型不支持

    Note:
        R² 的可能范围：
        - 1.0 表示完美预测
        - 0.0 表示预测等同于常数预测（用均值）
        - 负值表示预测比常数预测更差
        - 常数 y_true 时 R² 未定义（除零），抛 ValueError
    """
    yt = _to_numpy(y_true, "y_true")
    yp = _to_numpy(y_pred, "y_pred")
    _check_pair(yt, yp)

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))

    if ss_tot == 0:
        raise ValueError(
            "R² undefined: y_true has zero variance (constant target). "
            "This matches sklearn.metrics.r2_score behavior (raises ValueError)."
        )

    return float(1 - ss_res / ss_tot)


__all__ = ["mape", "mae", "rmse", "r2"]
