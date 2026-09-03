"""UI package for the China Industrial PPI dashboard (P0.11 redesign).

Pages are rendered by modules in this package; `app/streamlit_app.py` is a
thin router (data contract + sidebar + page dispatch). No research logic
lives here: all formal numbers come from `constants.py`, which mirrors the
locked records in docs/PROJECT_STATUS.md sections 4 and 5.

P0.11 page set (CN router labels -> module):
总览 overview / 数据 data_page / 趋势分析 trend / 预测 forecast /
模型评估 evaluation / 稳健性检验 robustness / 方法与说明 methodology.
Design: Research Analytics Terminal — dark navy chrome, white workspace,
restrained deep blue accent, Chinese-first copy with English terminal labels.
"""
