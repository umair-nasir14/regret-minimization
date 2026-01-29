from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .base import BaseGenerator
from .openrouter import OpenRouterLLM
from .utils import _json_default, _row_to_dict, _to_iso, _pretty_json, _escape_format_braces, _get_defaults


class ReasoningGenerator(BaseGenerator):
    """
    Renders `prompts/reasoninggenerator.prompt` and asks an LLM to explain the trade/action.
    """

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        super().__init__(prompt_path=prompt_path)
        self.data_path = Path(__file__).parent / "data" / "hindsight_optimal_trades_btc.parquet"
        self.response_prefix = "Reason"

    def variables(self, row: Any) -> Mapping[str, Any]:
       
        d = _row_to_dict(row)
        decision_ts = d.get("decision_timestamp")
        decision_ts_iso = _to_iso(decision_ts)
        state_json = d.get("state_json", "")
        state_pretty = _pretty_json(state_json)
        state_pretty = _escape_format_braces(state_pretty)

        return {
            "coin": _get_defaults(d, "coin"),
            "decision_timestamp_iso": decision_ts_iso,
            "action": _get_defaults(d, "action"),
            "no_trade_reason": _get_defaults(d, "no_trade_reason", ""),
            "close": _get_defaults(d, "close"),
            "ret_15m": _get_defaults(d, "ret_15m"),
            "rv_15m": _get_defaults(d, "rv_15m"),
            "funding": _get_defaults(d, "funding"),
            "funding_z_7d": _get_defaults(d, "funding_z_7d"),
            "ofi_1m": _get_defaults(d, "ofi_1m"),
            "impact_skew_bps": _get_defaults(d, "impact_skew_bps"),
            "open_interest": _get_defaults(d, "open_interest"),
            "hour": _get_defaults(d, "hour"),
            "dow": _get_defaults(d, "dow"),
            "is_gap_fill": _get_defaults(d, "is_gap_fill"),
            "trade_count": _get_defaults(d, "trade_count"),
            "gross_pnl_pct": _get_defaults(d, "gross_pnl_pct"),
            "net_pnl_pct": _get_defaults(d, "net_pnl_pct"),
            "entry_price": _get_defaults(d, "entry_price"),
            "ideal_exit_price": _get_defaults(d, "ideal_exit_price"),
            "time_to_peak_minutes": _get_defaults(d, "time_to_peak_minutes"),
            "max_adverse_excursion_pct": _get_defaults(d, "max_adverse_excursion_pct"),
            "long_potential_pct": _get_defaults(d, "long_potential_pct"),
            "short_potential_pct": _get_defaults(d, "short_potential_pct"),
            "direction_edge_pct": _get_defaults(d, "direction_edge_pct"),
            "state_json_pretty": state_pretty,
        }

def main():
    generator = ReasoningGenerator()
    generator.load_data()
    out_file = Path(__file__).parent / "results" / "reasonings.json"
    llm = OpenRouterLLM()

    for row_index in range(0, len(generator.data)):
        response = generator.generate(llm, row_index, temperature=0.7)
        generator.save(row_index=row_index, response_text=response, out_path=out_file)
        print(f"Appended row {row_index} -> {out_file}")

if __name__ == "__main__":
    main()

