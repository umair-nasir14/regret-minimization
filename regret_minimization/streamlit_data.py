from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RegretRow:
    row_index: int
    decision_timestamp: str
    coin: str
    hindsight_action: str
    classified_action: str
    analysis: str
    features: dict[str, Any]


def _results_dir() -> Path:
    # `research/regret_minimization/results/` sits next to this file.
    return Path(__file__).resolve().parent / "results"


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_json_loads(s: Any) -> Any | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _clean_analysis(s: Any) -> str:
    """
    Some rows contain analysis that is wrapped in a Markdown code fence, sometimes
    with content accidentally placed on the same line as the opening fence, e.g.:
      ```VERDICT: NOT_PREDICTABLE
      CONFIDENCE_IN_VERDICT: ...
      ...
      ```

    In Markdown, the text after ``` on the same line is treated as the "info string"
    (language), so "VERDICT: ..." won't render as content. This function removes an
    outermost opening/closing fence while preserving the meaningful first line.
    """
    if s is None:
        return ""

    t = str(s).replace("\r\n", "\n").replace("\r", "\n")
    # Occasionally content may include a BOM or leading whitespace/newlines.
    t = t.lstrip("\ufeff").lstrip()

    if t.startswith("```"):
        rest = t[3:]
        # If the fence line includes text, keep it unless it looks like a language tag.
        if "\n" in rest:
            first, remainder = rest.split("\n", 1)
            first_s = first.strip()
            # Heuristic: treat short non-colon tokens as language identifiers (e.g. "json").
            if first_s and (":" not in first_s) and (len(first_s) <= 15) and (" " not in first_s):
                t = remainder
            else:
                t = (first + "\n" + remainder).lstrip("\n")
        else:
            t = rest

    # Remove a trailing fence (allowing whitespace/newlines before it).
    t_r = t.rstrip()
    if t_r.endswith("```"):
        t = t_r[: -len("```")].rstrip("\n")

    return t


def load_regret_data(
    *,
    decompositions_path: Path | None = None,
    classifications_path: Path | None = None,
) -> pd.DataFrame:
    """
    Returns a merged dataframe with at least:
    - row_index (int)
    - features (dict)
    - Analysis (str)
    - Action (str)  # from action_classifications.json
    """
    results_dir = _results_dir()
    decompositions_path = decompositions_path or (results_dir / "hindsight_decompositions.json")
    classifications_path = classifications_path or (results_dir / "action_classifications.json")

    with decompositions_path.open("r", encoding="utf-8") as f:
        decomps = json.load(f)
    with classifications_path.open("r", encoding="utf-8") as f:
        classes = json.load(f)

    df_decomp = pd.DataFrame(decomps)
    df_class = pd.DataFrame(classes)

    if "row_index" not in df_decomp.columns:
        raise ValueError("Expected `row_index` in hindsight_decompositions.json rows.")
    if "features" not in df_decomp.columns:
        raise ValueError("Expected `features` in hindsight_decompositions.json rows.")

    if "row_index" not in df_class.columns:
        raise ValueError("Expected `row_index` in action_classifications.json rows.")
    if "Action" not in df_class.columns:
        raise ValueError("Expected `Action` in action_classifications.json rows.")

    df_decomp["row_index"] = df_decomp["row_index"].map(_safe_int)
    df_class["row_index"] = df_class["row_index"].map(_safe_int)
    df_decomp = df_decomp.dropna(subset=["row_index"]).copy()
    df_class = df_class.dropna(subset=["row_index"]).copy()

    df_decomp["row_index"] = df_decomp["row_index"].astype(int)
    df_class["row_index"] = df_class["row_index"].astype(int)

    # The files can be appended over time; dedupe by `row_index` and keep the latest.
    # (We assume later rows are more up-to-date.)
    df_decomp = df_decomp.drop_duplicates(subset=["row_index"], keep="last").copy()
    df_class = df_class.drop_duplicates(subset=["row_index"], keep="last").copy()

    # If `Action` exists in decompositions too, keep the classification as `Action` and rename hindsight-side.
    df = df_decomp.merge(
        df_class[["row_index", "Action"]],
        how="left",
        on="row_index",
        suffixes=("", "_classified"),
    )

    # Convenience columns for UI.
    def _h_action(features: Any) -> str:
        if not isinstance(features, dict):
            return ""
        a = features.get("action") or ""
        reason = features.get("no_trade_reason") or ""
        if reason and str(a).lower() in {"no_trade", "no-trade", "flat"}:
            return f"{a} ({reason})"
        return str(a)

    def _h_summary(features: Any) -> str:
        if not isinstance(features, dict):
            return _h_action(features)
        coin = features.get("coin") or ""
        ts = features.get("decision_timestamp") or features.get("timestamp_iso") or ""
        action = _h_action(features)
        parts = [p for p in [coin, ts, action] if p]
        return " — ".join(str(p) for p in parts) if parts else action

    df["hindsight_action"] = df["features"].map(_h_action)
    df["hindsight_summary"] = df["features"].map(_h_summary)

    def _coin(features: Any) -> str:
        if not isinstance(features, dict):
            return ""
        return str(features.get("coin") or "")

    df["coin"] = df["features"].map(_coin)

    def _decision_ts(features: Any) -> str:
        if not isinstance(features, dict):
            return ""
        return str(features.get("decision_timestamp") or features.get("timestamp_iso") or "")

    df["decision_timestamp"] = df["features"].map(_decision_ts)

    # Parse the embedded `state_json` if present (kept as separate column for display).
    def _parsed_state(features: Any) -> Any | None:
        if not isinstance(features, dict):
            return None
        return _safe_json_loads(features.get("state_json"))

    df["state_json_parsed"] = df["features"].map(_parsed_state)

    return df


def get_row(df: pd.DataFrame, row_index: int) -> RegretRow | None:
    if df.empty:
        return None
    sub = df.loc[df["row_index"] == int(row_index)]
    if sub.empty:
        return None
    r = sub.iloc[0].to_dict()
    features = r.get("features")
    if not isinstance(features, dict):
        features = {}
    return RegretRow(
        row_index=int(r.get("row_index")),
        decision_timestamp=str(r.get("decision_timestamp") or features.get("decision_timestamp") or features.get("timestamp_iso") or ""),
        coin=str(r.get("coin") or features.get("coin") or ""),
        hindsight_action=str(r.get("hindsight_action") or ""),
        classified_action=str(r.get("Action") or ""),
        analysis=_clean_analysis(r.get("Analysis") or ""),
        features=features,
    )

