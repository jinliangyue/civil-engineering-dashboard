"""UI package for the China Industrial PPI dashboard (P0.10 redesign).

Pages are rendered by modules in this package; `app/streamlit_app.py` is a
thin router (data contract + sidebar + page dispatch). No research logic
lives here: all formal numbers come from `constants.py`, which mirrors the
locked records in docs/PROJECT_STATUS.md sections 4 and 5.
"""
