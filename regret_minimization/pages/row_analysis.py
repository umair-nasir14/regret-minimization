from __future__ import annotations

import sys
import inspect
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

_COLUMNS_PARAMS = inspect.signature(st.columns).parameters


def _columns(spec, *, vertical_alignment: str | None = None):
    if vertical_alignment is not None and "vertical_alignment" in _COLUMNS_PARAMS:
        return st.columns(spec, vertical_alignment=vertical_alignment)
    return st.columns(spec)


def _qp_get(key: str) -> str | None:
    if hasattr(st, "query_params"):
        v = st.query_params.get(key)  # type: ignore[attr-defined]
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    try:
        d = st.experimental_get_query_params()
        vals = d.get(key)
        if not vals:
            return None
        s = str(vals[0]).strip()
        return s or None
    except Exception:
        return None


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


def _get_selected_row_index() -> int | None:
    qp = _qp_get("row")
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


df = _load_df(_get_results_signature())
row_index = _get_selected_row_index()

st.title("Reasoning Trace Analysis")

if row_index is None:
    st.warning("No row selected. Go back and click a row in the table.")
    if st.button("Back to table", type="primary"):
        if hasattr(st, "switch_page"):
            st.switch_page("streamlit_app.py")
    st.stop()

row = get_row(df, row_index=row_index)
if row is None:
    st.error(f"Row `{row_index}` not found in the loaded results.")
    if st.button("Back to table", type="primary"):
        if hasattr(st, "switch_page"):
            st.switch_page("streamlit_app.py")
    st.stop()

top = st.container()
with top:
    c1, c2, c3, c4 = _columns([1, 2, 1, 2], vertical_alignment="center")
    c1.metric("row_index", row.row_index)
    c2.metric("timestamp", row.decision_timestamp or "")
    c3.metric("LLM Classifier Actions", row.classified_action or "")
    c4.metric("Hindsight Optimal Action", row.hindsight_action or "")

st.divider()

left, right = _columns([3, 2], vertical_alignment="top")

with left:
    st.subheader("Reasoning Trace")
    # Use rstrip (not strip) so we never drop leading content/lines.
    analysis = (row.analysis or "").rstrip()
    if analysis:
        # Render as plain text (no Markdown parsing) to avoid code-fence edge cases on deployment.
        # text_area tends to be the most stable across Streamlit versions and clearly preserves all lines.
        st.text_area("Reasoning Trace", value=analysis, height=520)
    else:
        st.info("No `Reasoning Trace` field found for this row.")

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
    if hasattr(st, "switch_page"):
        st.switch_page("streamlit_app.py")

