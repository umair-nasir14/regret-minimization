from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd
from pathlib import Path


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a row to a dictionary."""
    if isinstance(row, dict):
        return row
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    # Best-effort fallback for Mapping-like objects
    if isinstance(row, Mapping):
        return dict(row)
    raise TypeError(f"Unsupported row type: {type(row)!r}. Expected dict/Mapping or pandas Series-like object.")


def _to_iso(ts: Any) -> str:
    """Convert a timestamp to an ISO string."""
    if ts is None:
        return ""
    # pandas.Timestamp has isoformat(); datetime does too.
    iso = getattr(ts, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(ts)


def _pretty_json(s: Any) -> str:
    """Convert a JSON string to a pretty string."""
    if not s:
        return ""
    if not isinstance(s, str):
        return str(s)
    try:
        obj = json.loads(s)
    except Exception:
        return s
    return json.dumps(obj, indent=2, sort_keys=True)

def _get_defaults(d: dict[str, Any], key: str, default: Any = "") -> Any:
    """Get a value from the dictionary with a default."""
    v = d.get(key, default)
    if v is None:
        return default
    return v


def _escape_format_braces(s: str) -> str:
    """Escape braces in a string."""
    return s.replace("{", "{{").replace("}", "}}")


def _json_default(obj: Any) -> Any:
    """
    Make common pandas/numpy objects JSON-serializable.
    - pandas.Timestamp -> ISO string
    - numpy scalars -> Python scalars via `.item()`
    - everything else -> `str(obj)`
    """
    iso = getattr(obj, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass

    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(obj)


def _load_df(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataframe from the data path.
    
    data_path: The path to the data file.
    Returns: The dataframe.
    """
    if data_path is None:
        raise ValueError("`data_path` is not set. Set `data_path` in the child class.") 
    path = Path(data_path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data file type: {path.suffix}. Supported: .parquet, .csv")

def _parse_response(response_text: str, response_prefix: str) -> str:
    """
    Parse the response text for the response prefix.
    
    response_text: The response text to parse.
    response_prefix: The response prefix to parse.
    Returns: The parsed response text.
    """
    if not response_text:
        return ""
    text = response_text.strip()
    prefix = (response_prefix or "").strip()
    prefix_l = prefix.lower()
    for line in text.splitlines():
        # Compare case-insensitively (models sometimes vary casing).
        if line.lstrip().lower().startswith(f"{prefix_l}:"):
            return line.split(":", 1)[1].strip()
    idx = text.lower().find(f"{prefix_l}:")
    if idx != -1:
        return text[idx + len(f"{prefix_l}:") :].strip()
    return text


def _parse_prompt_file(text: str) -> tuple[str | None, str]:
    """
    Parse a `.prompt` file with `System:` and `User:` sections.
    - `System:` is optional
    - `User:` is required
    """
    system_lines: list[str] = []
    user_lines: list[str] = []
    current: str | None = None
    for raw_line in text.splitlines():
        if raw_line.startswith("System:"):
            current = "system"
            system_lines.append(raw_line[len("System:") :].lstrip())
            continue
        if raw_line.startswith("User:"):
            current = "user"
            user_lines.append(raw_line[len("User:") :].lstrip())
            continue
        if current == "system":
            system_lines.append(raw_line)
        elif current == "user":
            user_lines.append(raw_line)
    user = "\n".join(user_lines).strip()
    if not user:
        raise ValueError("Prompt file missing required `User:` section.")
    system = "\n".join(system_lines).strip() if system_lines else None
    return system, user