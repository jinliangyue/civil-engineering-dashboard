"""
metrics.py 最小测试脚本

不引入 pytest 等测试框架，用简单 assert 验证关键边界情况。

覆盖：
1. 正常数据
2. 完全预测正确
3. MAPE 中存在 y_true=0
4. 全部 y_true=0（应该 raise ValueError）
5. 常量真实值的 R²（应该 raise ValueError）
6. list / numpy / pandas 三种输入
7. 长度不匹配（应该 raise ValueError）

运行方式：
    python3 -m src.evaluation.test_metrics

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""

import sys
import numpy as np
import pandas as pd

from src.evaluation.metrics import mape, mae, rmse, r2


def assert_close(actual, expected, tol=1e-9, msg=""):
    if abs(actual - expected) > tol:
        raise AssertionError(f"FAIL {msg}: expected {expected}, got {actual}")
    print(f"  PASS {msg}: {actual}")


def assert_raises(fn, exc_type, msg=""):
    try:
        fn()
    except exc_type as e:
        print(f"  PASS {msg}: raised {exc_type.__name__}: {str(e)[:60]}")
        return
    except Exception as e:
        raise AssertionError(
            f"FAIL {msg}: expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"FAIL {msg}: expected {exc_type.__name__} but no exception")


def test_normal():
    print("\n[Test 1] 正常数据")
    yt = [100.0, 110.0, 95.0, 105.0]
    yp = [102.0, 108.0, 97.0, 103.0]
    # MAE = mean(|100-102|, |110-108|, |95-97|, |105-103|) = mean(2,2,2,2) = 2.0
    assert_close(mae(yt, yp), 2.0, msg="MAE 正常")
    # RMSE = sqrt(mean(4,4,4,4)) = 2.0
    assert_close(rmse(yt, yp), 2.0, msg="RMSE 正常")
    # MAPE = mean(2/100, 2/110, 2/95, 2/105) * 100 ≈ 1.9436
    expected_mape = (2/100 + 2/110 + 2/95 + 2/105) / 4 * 100
    assert_close(mape(yt, yp), expected_mape, tol=1e-9, msg="MAPE 正常")
    # R² ≈ 0.969（线性相关性高）
    # SS_res = 4+4+4+4 = 16; SS_tot = sum((yt-mean(yt))^2)
    # yt mean = 102.5; (100-102.5)^2 + (110-102.5)^2 + (95-102.5)^2 + (105-102.5)^2
    # = 6.25 + 56.25 + 56.25 + 6.25 = 125
    # R² = 1 - 16/125 = 0.872
    assert_close(r2(yt, yp), 1 - 16/125, msg="R² 正常")


def test_perfect_prediction():
    print("\n[Test 2] 完全预测正确")
    yt = [98.0, 99.5, 101.2, 100.0, 99.8]
    yp = [98.0, 99.5, 101.2, 100.0, 99.8]
    assert_close(mae(yt, yp), 0.0, msg="MAE 完美预测")
    assert_close(rmse(yt, yp), 0.0, msg="RMSE 完美预测")
    assert_close(mape(yt, yp), 0.0, msg="MAPE 完美预测")
    assert_close(r2(yt, yp), 1.0, msg="R² 完美预测")


def test_mape_with_zero_true():
    print("\n[Test 3] MAPE 中存在 y_true=0（应跳过零值）")
    yt = [100.0, 0.0, 110.0]
    yp = [102.0, 5.0, 108.0]
    # MAPE 只算 yt != 0 的: |100-102|/100 + |110-108|/110 = 0.02 + 0.01818 = 0.03818
    # mean = 0.01909, * 100 = 1.909
    expected = ((2/100) + (2/110)) / 2 * 100
    assert_close(mape(yt, yp), expected, msg="MAPE 跳过零值")
    # MAE / RMSE 不受影响（包含全部 3 个点）
    # MAE = mean(2, 5, 2) = 3
    assert_close(mae(yt, yp), 3.0, msg="MAE 包含零值点")


def test_mape_all_zero_true():
    print("\n[Test 4] 全部 y_true=0（应 raise ValueError）")
    yt = [0.0, 0.0, 0.0]
    yp = [1.0, 2.0, 3.0]
    assert_raises(
        lambda: mape(yt, yp),
        ValueError,
        msg="MAPE 全部 y_true=0 应报错"
    )


def test_r2_constant_true():
    print("\n[Test 5] 常量真实值的 R²（应 raise ValueError）")
    yt = [100.0, 100.0, 100.0, 100.0]
    yp = [100.0, 101.0, 99.0, 100.0]
    assert_raises(
        lambda: r2(yt, yp),
        ValueError,
        msg="R² 常量 y_true 应报错"
    )


def test_input_types():
    print("\n[Test 6] 输入类型（list / numpy / pandas）")
    yt_list = [100.0, 110.0, 95.0]
    yp_list = [102.0, 108.0, 97.0]

    yt_np = np.array(yt_list)
    yp_np = np.array(yp_list)

    yt_pd = pd.Series(yt_list)
    yp_pd = pd.Series(yp_list)

    # list
    m1 = mape(yt_list, yp_list)
    # numpy
    m2 = mape(yt_np, yp_np)
    # pandas
    m3 = mape(yt_pd, yp_pd)

    assert_close(m1, m2, msg="list 与 numpy 一致")
    assert_close(m1, m3, msg="list 与 pandas 一致")

    # 混合输入：list vs numpy
    m4 = mape(yt_list, yp_np)
    assert_close(m1, m4, msg="list + numpy 混合输入")

    # MAE 同样
    assert_close(mae(yt_list, yp_list), mae(yt_np, yp_np), msg="MAE 类型一致")
    # RMSE
    assert_close(rmse(yt_list, yp_list), rmse(yt_pd, yp_pd), msg="RMSE 类型一致")
    # R²
    assert_close(r2(yt_list, yp_list), r2(yt_np, yp_pd), msg="R² 类型一致")


def test_length_mismatch():
    print("\n[Test 7] 长度不匹配（应 raise ValueError）")
    yt = [100.0, 110.0, 95.0]
    yp = [102.0, 108.0]
    assert_raises(lambda: mape(yt, yp), ValueError, msg="MAPE 长度不匹配")
    assert_raises(lambda: mae(yt, yp), ValueError, msg="MAE 长度不匹配")
    assert_raises(lambda: rmse(yt, yp), ValueError, msg="RMSE 长度不匹配")
    assert_raises(lambda: r2(yt, yp), ValueError, msg="R² 长度不匹配")


def test_empty_input():
    print("\n[Test 8] 空输入（应 raise ValueError）")
    assert_raises(lambda: mape([], []), ValueError, msg="MAPE 空输入")
    assert_raises(lambda: mae([], []), ValueError, msg="MAE 空输入")
    assert_raises(lambda: rmse([], []), ValueError, msg="RMSE 空输入")
    assert_raises(lambda: r2([], []), ValueError, msg="R² 空输入")


def test_invalid_type():
    print("\n[Test 9] 不支持的输入类型（应 raise TypeError）")
    assert_raises(lambda: mape("not_array", [1.0, 2.0]), TypeError, msg="MAPE 字符串输入")
    assert_raises(lambda: mae(123, [1.0, 2.0]), TypeError, msg="MAE 整数输入")


def main():
    print("=" * 60)
    print("metrics.py 单元测试")
    print("=" * 60)
    test_normal()
    test_perfect_prediction()
    test_mape_with_zero_true()
    test_mape_all_zero_true()
    test_r2_constant_true()
    test_input_types()
    test_length_mismatch()
    test_empty_input()
    test_invalid_type()
    print("\n" + "=" * 60)
    print("全部测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
