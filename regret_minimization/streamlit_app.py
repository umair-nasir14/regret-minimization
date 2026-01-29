from __future__ import annotations

import sys
import inspect
from pathlib import Path

import streamlit as st

# Streamlit may not include the repo root on `sys.path`, so import locally.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from streamlit_data import load_regret_data  # noqa: E402


st.set_page_config(
    page_title="Regret Minimization – Results Browser",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_df(_results_signature: tuple[tuple[str, int, int], ...]):
    return load_regret_data()

def _get_results_signature() -> tuple[tuple[str, int, int], ...]:
    results_dir = _APP_DIR / "results"
    candidates = [
        results_dir / "hindsight_decompositions.json",
        results_dir / "hindsight_decompositions_old.json",
        results_dir / "action_classifications.json",
        results_dir / "action_classification.json",
        results_dir / "action_classifications_old.json",
    ]
    sig: list[tuple[str, int, int]] = []
    for p in candidates:
        if p.exists():
            s = p.stat()
            sig.append((p.name, int(getattr(s, "st_mtime_ns", int(s.st_mtime * 1e9))), int(s.st_size)))
    return tuple(sig)


st.title("Regret Minimization – Results Browser")
st.caption("Browse hindsight decompositions vs. classified actions. Click a row to open its detailed analysis.")

df = _load_df(_get_results_signature())

total_rows = int(df["row_index"].nunique()) if "row_index" in df.columns else int(len(df))
if "Analysis" in df.columns:
    _a = df["Analysis"]
    rows_with_analysis = int((_a.notna() & _a.astype(str).str.strip().ne("")).sum())
else:
    rows_with_analysis = 0

c1, c2 = st.columns([1, 5], vertical_alignment="center")
c1.metric("Rows w/ reasoning traces", rows_with_analysis)
c2.caption(f"Total rows: {total_rows}")

view = df[["row_index", "decision_timestamp", "coin", "hindsight_action", "Action"]].copy()
view = view.rename(
    columns={
        "decision_timestamp": "timestamp",
        "coin": "coin",
        "hindsight_action": "Hindsight Action",
        "Action": "Action",
    }
)

display_df = view[["timestamp", "coin", "Hindsight Action", "Action"]].copy()
display_df = display_df.set_index("timestamp")

df_params = inspect.signature(st.dataframe).parameters
supports_row_selection = "selection_mode" in df_params and "on_select" in df_params

if supports_row_selection:
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        selection_mode="single-row",
        on_select="rerun",
        key="results_table",
        column_config={
            "timestamp": st.column_config.TextColumn("timestamp", width="medium"),
            "coin": st.column_config.TextColumn("coin", width="small"),
            "Hindsight Action": st.column_config.TextColumn("Hindsight Action", width="large"),
            "Action": st.column_config.TextColumn("Action", width="small"),
        },
    )

    sel = getattr(event, "selection", None)
    sel_rows = getattr(sel, "rows", None) if sel is not None else None

    if sel_rows:
        selected = view.iloc[int(sel_rows[0])]
        selected_row_index = int(selected["row_index"])
        st.session_state["selected_row_index"] = selected_row_index
        st.query_params["row"] = str(selected_row_index)
        st.switch_page("pages/row_analysis.py")
else:
    st.info(
        "Your Streamlit version doesn’t support clickable row selection in tables. "
        "Use the selector below to open a row’s details."
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            "timestamp": st.column_config.TextColumn("timestamp", width="medium"),
            "coin": st.column_config.TextColumn("coin", width="small"),
            "Hindsight Action": st.column_config.TextColumn("Hindsight Action", width="large"),
            "Action": st.column_config.TextColumn("Action", width="small"),
        },
    )

    options = [
        f"{r['timestamp']}  |  {r['coin']}  |  H={r['Hindsight Action']}  |  A={r['Action']}"
        for _, r in view.iterrows()
    ]
    default_i = 0
    last_row = st.session_state.get("selected_row_index")
    if last_row is not None:
        try:
            last_row = int(last_row)
            match = view.index[view["row_index"] == last_row]
            if len(match) > 0:
                default_i = int(match[0])
        except Exception:
            pass

    picked = st.selectbox("Open by timestamp", options=options, index=default_i)
    picked_i = int(options.index(picked))
    selected_row_index = int(view.iloc[picked_i]["row_index"])

    if st.button("Open analysis", type="primary"):
        st.session_state["selected_row_index"] = selected_row_index
        st.query_params["row"] = str(selected_row_index)
        st.switch_page("pages/row_analysis.py")

st.divider()


