"""
项目评估指标统一模块

集中定义项目后续实验使用的：
- MAPE
- MAE
- RMSE
- R²

所有函数：
- 接受 list / numpy.ndarray / pandas.Series 输入
- 不静默产生 inf / nan
- 边界情况抛 ValueError 并说明原因
- R² 与 sklearn.metrics.r2_score 语义一致

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""
