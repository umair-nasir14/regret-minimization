from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from sklearn.metrics import accuracy_score, f1_score

from .base import BaseGenerator
from .openrouter import OpenRouterLLM
from .utils import _escape_format_braces, _get_defaults, _parse_response, _row_to_dict, _to_iso


_ALLOWED_ACTIONS = ("long", "short", "no_trade")
_ACTION_TO_LABEL: dict[str, int] = {"long": 0, "short": 1, "no_trade": 2}
_LABEL_TO_ACTION: dict[int, str] = {v: k for k, v in _ACTION_TO_LABEL.items()}


def _encode_action(v: Any) -> int:
    if v is None:
        return -1
    s = str(v).strip().lower()
    # tolerate a couple common variants
    if s in {"no trade", "notrade", "no-trade", "none"}:
        return _ACTION_TO_LABEL["no_trade"]
    return _ACTION_TO_LABEL.get(s, -1)


def _decode_action(label: Any) -> str:
    try:
        i = int(label)
    except Exception:
        return ""
    return _LABEL_TO_ACTION.get(i, "")


def _explain_state_feature(key: str) -> str:
    """
    Very brief, non-speculative explanation for common `state_json` keys.
    Falls back to a generic description when unknown.
    """
    k = key.strip().lower()
    mapping = {
        "open": "open price for the bar",
        "high": "high price for the bar",
        "low": "low price for the bar",
        "close": "close price for the bar",
        "volume": "traded volume for the bar",
        "trade_count": "number of trades in the bar",
        "ret_1m": "return over the last 1 minute",
        "ret_5m": "return over the last 5 minutes",
        "ret_15m": "return over the last 15 minutes",
        "ret_60m": "return over the last 60 minutes",
        "ret_240m": "return over the last 240 minutes",
        "rv_15m": "realized volatility over the last 15 minutes",
        "rv_60m": "realized volatility over the last 60 minutes",
        "rv_240m": "realized volatility over the last 240 minutes",
        "range_bps": "bar high-low range in basis points",
        "body_bps": "candle body size in basis points",
        "upper_wick_bps": "upper wick size in basis points",
        "lower_wick_bps": "lower wick size in basis points",
        "close_loc": "close location within the high-low range",
        "log_vol": "log-transformed volume",
        "vol_z_60m": "volume z-score (60m window)",
        "trade_count_z_60m": "trade count z-score (60m window)",
        "trend_strength_60m": "trend strength feature (60m horizon)",
        "is_gap_fill": "flag for gap-fill pattern",
        "funding": "perp funding rate (if available)",
        "funding_z_1d": "funding z-score (1d window)",
        "funding_z_7d": "funding z-score (7d window)",
        "open_interest": "open interest (if available)",
        "oi_log": "log(open interest) (if available)",
        "oi_chg_15m": "open interest change over 15m",
        "oi_chg_60m": "open interest change over 60m",
        "oi_chg_240m": "open interest change over 240m",
        "oi_z_7d": "open interest z-score (7d window)",
        "ofi_1m": "order flow imbalance over 1 minute",
        "signed_vol": "signed volume proxy",
        "buy_vol": "buy-side volume",
        "sell_vol": "sell-side volume",
        "liq_vol": "liquidation volume (if available)",
        "impact_spread_bps": "impact spread in basis points",
        "impact_skew_bps": "impact skew in basis points",
        "impact_skew_z_60m": "impact skew z-score (60m window)",
        "d_impact_skew_5m": "change in impact skew over 5m",
        "premium": "mark/oracle premium (if available)",
        "mid_price": "mid price (if available)",
        "mark_price": "mark price (if available)",
        "oracle_price": "oracle/index price (if available)",
        "timestamp_iso": "timestamp of the state snapshot",
        "hour": "hour of day",
        "minute_of_day": "minute index within the day",
        "dow": "day of week",
        "is_weekend": "weekend flag",
        "day_of_month": "day of month",
        "month": "month number",
        "is_asia_session": "Asia session flag",
        "is_europe_session": "Europe session flag",
        "is_us_session": "US session flag",
        "hours_until_funding": "time until next funding event",
        "is_near_funding": "flag for proximity to funding",
        "augmented_by_l2": "flag indicating L2 augmentation",
        "ask_leg_bps": "ask-side depth/impact leg (bps)",
        "bid_leg_bps": "bid-side depth/impact leg (bps)",
    }
    return mapping.get(k, "feature from the state snapshot")


def _format_state_json_features(state_json: Any, *, max_items: int = 80) -> str:
    """
    Convert `state_json` (JSON string or dict-like) into bullet lines:
    - key = value - brief explanation
    """
    if not state_json:
        return "- (missing) - no state_json provided"

    obj: Any
    if isinstance(state_json, str):
        try:
            obj = json.loads(state_json)
        except Exception:
            return f"- (unparseable) - state_json was not valid JSON: {state_json!r}"
    else:
        obj = state_json

    if not isinstance(obj, dict):
        return f"- (unexpected) - state_json parsed to {type(obj)!r}, expected an object/dict"

    keys = sorted(obj.keys(), key=lambda x: str(x))
    lines: list[str] = []
    for i, k in enumerate(keys):
        if i >= max_items:
            lines.append(f"- ... ({len(keys) - max_items} more keys omitted)")
            break
        v = obj.get(k)
        explanation = _explain_state_feature(str(k))
        lines.append(f"- {k} = {v} - {explanation}")
    return "\n".join(lines)


class ActionClassifier(BaseGenerator):
    """
    Renders `prompts/actionclassifier.prompt` and asks an LLM to predict the `action`
    label: one of {long, short, no_trade}.

    Also provides `compute_metrics(...)` for accuracy and macro-F1 evaluation.
    """

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        super().__init__(prompt_path=prompt_path)
        self.data_path = Path(__file__).parent / "data" / "hindsight_optimal_trades_btc.parquet"
        self.response_prefix = "Action"

    def variables(self, row: Any) -> Mapping[str, Any]:
        d = _row_to_dict(row)
        decision_ts_iso = _to_iso(d.get("decision_timestamp"))

        state_features = _format_state_json_features(d.get("state_json", ""), max_items=120)
        # Important: prompt templates are `.format(...)`, so escape braces if present
        state_features = _escape_format_braces(state_features)

        return {
            "coin": _get_defaults(d, "coin"),
            "decision_timestamp_iso": decision_ts_iso,
            # state_json-derived features (listed one-by-one with brief explanation)
            "state_features": state_features,
            # trade-related (top-level) features
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

    def compute_metrics(self, y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
        yt = [int(x) for x in y_true]
        yp = [int(x) for x in y_pred]
        labels = list(_ACTION_TO_LABEL.values())
        return {
            "n": len(yt),
            "accuracy": float(accuracy_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "label_mapping": dict(_ACTION_TO_LABEL),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM action classifier for hindsight-optimal trades.")
    parser.add_argument("--start", type=int, default=0, help="Row index to start from (default: 0).")
    parser.add_argument("--max-rows", type=int, default=None, help="Max number of rows to process (default: all).")
    parser.add_argument("--temperature", type=float, default=0.5, help="LLM temperature (default: 0.0).")
    args = parser.parse_args()

    generator = ActionClassifier()
    generator.load_data()
    out_file = Path(__file__).parent / "results" / "action_classifications.json"
    metrics_file = Path(__file__).parent / "results" / "action_classification_metrics.json"
    llm = OpenRouterLLM()

    y_true: list[int] = []
    y_pred: list[int] = []

    start = max(0, int(args.start))
    end = len(generator.data)
    if args.max_rows is not None:
        end = min(end, start + int(args.max_rows))

    for row_index in range(start, end):
        row = generator.get_row(row_index)
        y_true.append(_encode_action(row.get("action")))

        response = generator.generate(llm, row_index, temperature=float(args.temperature))
        generator.save(row_index=row_index, response_text=response, out_path=out_file)

        # Parse the model response consistently with the saver (look for "Action:" prefix).
        pred = _parse_response(response, generator.response_prefix)
        y_pred.append(_encode_action(pred))

        if (row_index + 1) % 50 == 0:
            m = generator.compute_metrics(y_true, y_pred)
            print(f"[{row_index + 1}/{end}] accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")

    metrics = generator.compute_metrics(y_true, y_pred)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved predictions -> {out_file}")
    print(f"Saved metrics -> {metrics_file}")
    print(f"Final accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

