from __future__ import annotations

import sys
import inspect
from pathlib import Path

import streamlit as st

# Streamlit may not include the repo root on `sys.path`, so import locally.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from streamlit_data import get_row, load_regret_data  # noqa: E402

try:
    # Optional: enables true row-click selection on older Streamlit versions.
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode  # type: ignore

    _HAS_AGGRID = True
except Exception:
    _HAS_AGGRID = False


st.set_page_config(
    page_title="Regret Minimization – Results Browser",
    layout="wide",
)

_COLUMNS_PARAMS = inspect.signature(st.columns).parameters


def _columns(spec, *, vertical_alignment: str | None = None):
    """
    Streamlit `st.columns(..., vertical_alignment=...)` was added after some versions.
    Keep compatibility by only passing it when supported.
    """
    if vertical_alignment is not None and "vertical_alignment" in _COLUMNS_PARAMS:
        return st.columns(spec, vertical_alignment=vertical_alignment)
    return st.columns(spec)


def _qp_get(key: str) -> str | None:
    # Streamlit >= 1.30-ish has `st.query_params`; older versions use experimental APIs.
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


def _qp_set(**kwargs: str | int | None) -> None:
    payload = {k: str(v) for k, v in kwargs.items() if v is not None and str(v).strip() != ""}
    if hasattr(st, "query_params"):
        # Update only provided keys
        for k, v in payload.items():
            st.query_params[k] = v  # type: ignore[attr-defined]
        return

    try:
        st.experimental_set_query_params(**payload)
    except Exception:
        pass


def _switch_to_row(row_index: int) -> None:
    st.session_state["selected_row_index"] = int(row_index)
    _qp_set(row=int(row_index))


def _render_row_analysis_inline(df, *, row_index: int) -> None:
    row = get_row(df, row_index=row_index)
    if row is None:
        st.error(f"Row `{row_index}` not found in the loaded results.")
        return

    top = st.container()
    with top:
        c1, c2, c3, c4 = _columns([1, 2, 1, 2], vertical_alignment="center")
        c1.metric("row_index", row.row_index)
        c2.metric("timestamp", row.decision_timestamp or "")
        c3.metric("Action (classified)", row.classified_action or "")
        c4.metric("Hindsight Action", row.hindsight_action or "")

    st.divider()

    left, right = _columns([3, 2], vertical_alignment="top")

    with left:
        st.subheader("Analysis")
        analysis = (row.analysis or "").strip()
        if analysis:
            # Render as plain text (no Markdown parsing) to avoid code-fence edge cases on deployment.
            st.code(analysis, language="text")
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
st.caption("Click a row to open the reasoning traces.")

df = _load_df(_get_results_signature())

total_rows = int(df["row_index"].nunique()) if "row_index" in df.columns else int(len(df))
if "Analysis" in df.columns:
    _a = df["Analysis"]
    rows_with_analysis = int((_a.notna() & _a.astype(str).str.strip().ne("")).sum())
else:
    rows_with_analysis = 0

c1, c2 = _columns([1, 5], vertical_alignment="center")
c1.metric("Rows w/ reasoning traces", rows_with_analysis)
c2.caption(f"Total rows: {total_rows}")

view = df[["row_index", "decision_timestamp", "coin", "hindsight_action", "Action"]].copy()
view = view.rename(
    columns={
        "decision_timestamp": "timestamp",
        "coin": "coin",
        "hindsight_action": "Hindsight Optimal Action",
        "Action": "LLM Classifier Actions",
    }
)

display_df = view[["timestamp", "coin", "Hindsight Optimal Action", "LLM Classifier Actions"]].copy()
display_df = display_df.set_index("timestamp")

df_params = inspect.signature(st.dataframe).parameters
supports_row_selection = "selection_mode" in df_params and "on_select" in df_params

selected_from_qp = _qp_get("row")
selected_row_index: int | None = None
if selected_from_qp is not None:
    try:
        selected_row_index = int(selected_from_qp)
    except Exception:
        selected_row_index = None
if selected_row_index is None:
    v = st.session_state.get("selected_row_index")
    if v is not None:
        try:
            selected_row_index = int(v)
        except Exception:
            selected_row_index = None

if supports_row_selection:
    # Streamlit's built-in row selection UI includes a checkbox column. In practice, users
    # can click anywhere on the row to select it, so we hide the checkbox to make the
    # interaction feel like "click the row to open".
    st.markdown(
        """
        <style>
        /* Scope to the dataframe widget */
        div[data-testid="stDataFrame"] [data-testid="baseCheckbox"] { display: none !important; }
        div[data-testid="stDataFrame"] input[type="checkbox"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
            "Hindsight Optimal Action": st.column_config.TextColumn("Hindsight Optimal Action", width="large"),
            "LLM Classifier Actions": st.column_config.TextColumn("LLM Classifier Actions", width="small"),
        },
    )

    sel = getattr(event, "selection", None)
    sel_rows = getattr(sel, "rows", None) if sel is not None else None

    if sel_rows:
        selected = view.iloc[int(sel_rows[0])]
        selected_row_index = int(selected["row_index"])
        _switch_to_row(selected_row_index)

        # Newer Streamlit multipage: switch if available, otherwise render inline.
        if hasattr(st, "switch_page"):
            st.switch_page("pages/row_analysis.py")
        else:
            st.session_state["show_inline_analysis"] = True
else:
    if _HAS_AGGRID:
        st.caption("Row click selection is enabled via AgGrid for this Streamlit version.")
        # Use a copy that includes row_index for selection -> analysis lookup.
        grid_df = view[["row_index", "timestamp", "coin", "Hindsight Optimal Action", "LLM Classifier Actions"]].copy()

        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=True)
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_column("row_index", hide=True)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        grid_options = gb.build()

        grid_resp = AgGrid(
            grid_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=False,
            fit_columns_on_grid_load=True,
            theme="streamlit",
            height=420,
        )

        selected_rows = (grid_resp or {}).get("selected_rows") or []
        if selected_rows:
            try:
                selected_row_index = int(selected_rows[0].get("row_index"))
                _switch_to_row(selected_row_index)
                st.session_state["show_inline_analysis"] = True
            except Exception:
                pass
    else:
        st.info(
            "This Streamlit version doesn’t support row-click events for `st.dataframe`. "
            "Install `streamlit-aggrid` to enable row-click selection, or use the selector below."
        )
        st.dataframe(display_df, use_container_width=True, hide_index=False)

        options = [
            f"{r['timestamp']}  |  {r['coin']}  |  H={r['Hindsight Optimal Action']}  |  A={r['LLM Classifier Actions']}"
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
            _switch_to_row(selected_row_index)
            st.session_state["show_inline_analysis"] = True

st.divider()

if st.session_state.get("show_inline_analysis") and selected_row_index is not None:
    st.subheader("Row analysis")
    if st.button("Back to table", type="primary"):
        st.session_state["show_inline_analysis"] = False
        # Keep selection in session/query params, but collapse analysis panel.
    _render_row_analysis_inline(df, row_index=selected_row_index)
    st.divider()


