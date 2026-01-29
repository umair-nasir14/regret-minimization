from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .base import BaseGenerator
from .openrouter import OpenRouterLLM
from .utils import _get_defaults, _row_to_dict, _to_iso


class HindsightDecomposer(BaseGenerator):
    """
    Renders `prompts/hindsightdecomposer.prompt` and asks an LLM to judge whether a
    hindsight-optimal trade was predictable from t=0 features alone.
    """

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        super().__init__(prompt_path=prompt_path)
        self.data_path = Path(__file__).parent / "data" / "hindsight_optimal_trades_btc.parquet"
        # We want to save the full structured output; pick a prefix that should not appear.
        self.response_prefix = "HindsightDecomposition"

    def variables(self, row: Any) -> Mapping[str, Any]:
        d = _row_to_dict(row)
        decision_ts_iso = _to_iso(d.get("decision_timestamp"))

        return {
            "coin": _get_defaults(d, "coin"),
            "decision_timestamp_iso": decision_ts_iso,
            "action": _get_defaults(d, "action"),
            "entry_price": _get_defaults(d, "entry_price"),
            "ideal_exit_price": _get_defaults(d, "ideal_exit_price"),
            "gross_pnl_pct": _get_defaults(d, "gross_pnl_pct"),
            "net_pnl_pct": _get_defaults(d, "net_pnl_pct"),
            "time_to_peak_minutes": _get_defaults(d, "time_to_peak_minutes"),
            "max_adverse_excursion_pct": _get_defaults(d, "max_adverse_excursion_pct"),
            "long_potential_pct": _get_defaults(d, "long_potential_pct"),
            "short_potential_pct": _get_defaults(d, "short_potential_pct"),
            "direction_edge_pct": _get_defaults(d, "direction_edge_pct"),
            "no_trade_reason": _get_defaults(d, "no_trade_reason", ""),
            # t=0 features (top-level columns)
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
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM hindsight decomposer for hindsight-optimal trades.")
    parser.add_argument("--start", type=int, default=0, help="Row index to start from (default: 0).")
    parser.add_argument("--max-rows", type=int, default=None, help="Max number of rows to process (default: all).")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (default: 0.2).")
    args = parser.parse_args()

    decomposer = HindsightDecomposer()
    decomposer.load_data()
    out_file = Path(__file__).parent / "results" / "hindsight_decompositions.json"
    llm = OpenRouterLLM()

    start = max(0, int(args.start))
    end = len(decomposer.data)
    if args.max_rows is not None:
        end = min(end, start + int(args.max_rows))

    for row_index in range(start, end):
        response = decomposer.generate(llm, row_index, temperature=float(args.temperature))
        decomposer.save(row_index=row_index, response_text=response, out_path=out_file)
        print(f"Appended row {row_index} -> {out_file}")


if __name__ == "__main__":
    main()

