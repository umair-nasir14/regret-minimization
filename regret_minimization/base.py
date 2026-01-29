from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

import json
import os

from .utils import _json_default, _load_df, _parse_response, _parse_prompt_file

class BaseLLM(ABC):
    """Base class for an LLM text completion call."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        key_env_var: str = "LLM_API_KEY",
        model: str = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> None:
        
        self.key_env_var = key_env_var
        self.model = model
        self.api_key = api_key or os.getenv(self.key_env_var)
        if not self.api_key:
            raise ValueError(f"API key not found for {self.model}. Set {self.key_env_var} or pass api_key=...")
        
        self.headers: dict[str, str] = {}
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if app_name:
            self.headers["X-Title"] = app_name
        self._client = None


    @abstractmethod
    def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> str:
        """
        Return the model's text response for a single prompt string.

        kwargs are passed through to the provider implementation (e.g. temperature, max_tokens).
        """


class BaseGenerator(ABC):
    """
    Base class for prompt-based generators stored in a sibling `.prompt` file.

    The `.prompt` file is expected to be in this format:

    System: ...
    User: ...

    Subclasses must implement `variables(row)` to supply template variables for
    formatting a single database row at a time.
    """

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        self._prompt_path = Path(prompt_path) if prompt_path is not None else self._default_prompt_path()
        raw = self._prompt_path.read_text(encoding="utf-8")
        self._system_template, self._user_template = _parse_prompt_file(raw)
        self.data_path: str | Path | None = None
        self.data = None
        self.response_prefix: str = "Response"

    @abstractmethod
    def variables(self, row: Any) -> Mapping[str, Any]:
        """Return variables used to format the `.prompt` template for this row."""

    def load_data(self) -> None:
        """Load the data from the data path."""
        if self.data is None:
            self.data = _load_df(self.data_path)

    def get_row(self, idx: int = 0) -> dict[str, Any]:
        """
        Returns one row as a plain dict.

        Supports `.parquet` and `.csv`.
        """
        if self.data is None:
            self.data = _load_df(self.data_path)
        return dict(self.data.iloc[idx].to_dict())

    def generate(self, llm: BaseLLM, row_index: int, **kwargs: Any) -> str:
        """
        Call the LLM (and optionally post-process) for a single row.

        `kwargs` are forwarded to the LLM call (e.g. temperature, max_tokens).
        """
        row = self.get_row(row_index)
        system, user = self.render_prompt(row)
        return llm.complete(user, system=system, **kwargs)

    def save(
        self,
        *,
        out_path: str | Path,
        row_index: int,
        response_text: str,
        append: bool = True,
    ) -> Path:
        """
        Minimal "do everything" JSON saver for multiple rows.

        - Builds an entry: {"row_index": ..., "features": ..., f"{self.response_prefix}": ...}
        - Ensures the output file is a JSON list
        - Appends (default) or overwrites with a single-item list
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # `_parse_response` returns the content *after* `"{prefix}:"` when present.
        parsed = _parse_response(response_text, self.response_prefix).strip()
        entry = {
            "row_index": row_index,
            "features": self.get_row(row_index),
            f"{self.response_prefix}": parsed,
        }
        payload: list[dict[str, Any]]
        if append and out_path.exists():
            existing_text = out_path.read_text(encoding="utf-8").strip()
            if existing_text:
                existing = json.loads(existing_text)
                if isinstance(existing, list):
                    payload = existing
                elif isinstance(existing, dict):
                    payload = [existing]
                else:
                    payload = []
            else:
                payload = []
            payload.append(entry)
        else:
            payload = [entry]
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return out_path

    def render_prompt(self, row: Any) -> tuple[str | None, str]:
        """Render `(system, user)` for a single row. `system` may be None.
        
        row: The row to render the prompt for.
        """
        vars_dict = dict(self.variables(row))
        system = self._system_template.format(**vars_dict) if self._system_template is not None else None
        user = self._user_template.format(**vars_dict)
        return system, user

    def _default_prompt_path(self) -> Path:
        """
        Default prompt path: `<subclass_module_dir>/<subclass_name>.prompt`.
        Example: class `RegretPrompt` -> `prompts/regretprompt.prompt`
        """
        # Resolve the actual defining module (works for packages).
        # When executed as a script, `__module__` may be "__main__", which also works.
        module = importlib.import_module(self.__class__.__module__)
        module_path = Path(getattr(module, "__file__", None) or __file__)
        return module_path.parent / f"prompts/{self.__class__.__name__.lower()}.prompt"

    

