from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit pages are executed with a limited `sys.path`; ensure the app dir is importable.
_APP_DIR = Path(__file__).resolve().parents[1]  # .../research/regret_minimization
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from streamlit_data import get_row, load_regret_data  # noqa: E402


st.set_page_config(
    page_title="Row Analysis",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_df():
    return load_regret_data()


def _get_selected_row_index() -> int | None:
    qp = st.query_params.get("row")
    if qp is not None and str(qp).strip() != "":
        try:
            return int(str(qp).strip())
        except Exception:
            return None
    v = st.session_state.get("selected_row_index")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


df = _load_df()
row_index = _get_selected_row_index()

st.title("Row Analysis")

if row_index is None:
    st.warning("No row selected. Go back and click a row in the table.")
    if st.button("Back to table", type="primary"):
        st.switch_page("streamlit_app.py")
    st.stop()

row = get_row(df, row_index=row_index)
if row is None:
    st.error(f"Row `{row_index}` not found in the loaded results.")
    if st.button("Back to table", type="primary"):
        st.switch_page("streamlit_app.py")
    st.stop()

top = st.container()
with top:
    c1, c2, c3, c4 = st.columns([1, 2, 1, 2])
    c1.metric("row_index", row.row_index)
    c2.metric("timestamp", row.decision_timestamp or "")
    c3.metric("Action (classified)", row.classified_action or "")
    c4.metric("Hindsight Action", row.hindsight_action or "")

st.divider()

left, right = st.columns([3, 2], vertical_alignment="top")

with left:
    st.subheader("Analysis")
    analysis = (row.analysis or "").strip()
    if analysis:
        # Preserve line breaks without forcing monospace unless you prefer st.code.
        st.markdown(analysis.replace("\n", "  \n"))
    else:
        st.info("No `Analysis` field found for this row.")

with right:
    st.subheader("Features")
    st.json(row.features, expanded=False)

    parsed = df.loc[df["row_index"] == row.row_index, "state_json_parsed"]
    parsed_obj = None
    if not parsed.empty:
        parsed_obj = parsed.iloc[0]
    if parsed_obj is not None:
        with st.expander("state_json (parsed)", expanded=False):
            st.json(parsed_obj, expanded=False)

st.divider()
if st.button("Back to table", type="primary"):
    st.switch_page("streamlit_app.py")

