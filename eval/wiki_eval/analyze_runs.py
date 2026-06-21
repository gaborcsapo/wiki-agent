"""Compare two FRAMES .eval logs (baseline vs feature) on efficiency metrics.

    uv run python -m wiki_eval.analyze_runs <baseline.eval> <feature.eval>

The agent calls Anthropic directly (bypassing Inspect's model layer), so token
usage is approximated from the trajectory and `avg_steps` (Sonnet round-trips)
is the reliable efficiency metric. The pure aggregator `_summarize_samples` is
unit-tested; log reading is thin I/O.
"""

from __future__ import annotations

import sys
from statistics import mean

CORRECT = "C"  # Inspect Score value for a correct model_graded_qa judgement


def _summarize_samples(records: list[dict]) -> dict:
    out = {"n": len(records), "accuracy": 0.0, "avg_steps": 0.0, "avg_tool_calls": 0.0,
           "total_tool_calls": 0, "batched_calls": 0, "batch_usage_rate": 0.0,
           "avg_batch_size": 0.0, "approx_tokens": 0}
    if not records:
        return out
    tool_inputs = [ti for r in records for ti in r["tool_inputs"]]
    batched = [ti for ti in tool_inputs if ti.get("queries") or ti.get("titles")]
    sizes = [len(ti.get("queries") or ti.get("titles")) for ti in batched]
    steps = [r["steps"] for r in records if r.get("steps") is not None]
    out.update(
        accuracy=sum(1 for r in records if r["correct"]) / len(records),
        avg_steps=mean(steps) if steps else 0.0,
        avg_tool_calls=mean([len(r["tool_inputs"]) for r in records]),
        total_tool_calls=len(tool_inputs),
        batched_calls=len(batched),
        batch_usage_rate=(len(batched) / len(tool_inputs)) if tool_inputs else 0.0,
        avg_batch_size=mean(sizes) if sizes else 0.0,
        approx_tokens=sum(r.get("tokens", 0) for r in records),
    )
    return out


def _record(sample) -> dict:
    """Flatten an Inspect EvalSample into the fields we aggregate on."""
    scores = getattr(sample, "scores", None) or {}
    mgqa = scores.get("model_graded_qa")
    correct = bool(mgqa is not None and getattr(mgqa, "value", None) == CORRECT)
    meta = getattr(sample, "metadata", None) or {}
    steps_list = (meta.get("trajectory") or {}).get("steps", [])
    tool_inputs = [s.get("tool_input") or {} for s in steps_list if s.get("kind") == "tool_call"]
    tokens = sum((s.get("input_tokens") or 0) + (s.get("output_tokens") or 0) for s in steps_list)
    return {"correct": correct, "steps": meta.get("steps"), "tool_inputs": tool_inputs, "tokens": tokens}


def _wall_clock(log) -> float | None:
    from datetime import datetime
    st = getattr(log.stats, "started_at", None)
    en = getattr(log.stats, "completed_at", None)
    if not st or not en:
        return None
    return (datetime.fromisoformat(en) - datetime.fromisoformat(st)).total_seconds()


def summarize_log(path: str) -> dict:
    from inspect_ai.log import read_eval_log
    log = read_eval_log(path)
    summary = _summarize_samples([_record(s) for s in (log.samples or [])])
    summary["wall_clock_s"] = _wall_clock(log)
    return summary


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv: list[str]) -> None:
    baseline, feature = summarize_log(argv[1]), summarize_log(argv[2])
    keys = ["n", "accuracy", "avg_steps", "avg_tool_calls", "total_tool_calls",
            "batched_calls", "batch_usage_rate", "avg_batch_size", "approx_tokens",
            "wall_clock_s"]
    print(f"{'metric':<18} {'baseline':>12} {'feature':>12} {'delta':>12}")
    print("-" * 56)
    for k in keys:
        b, f = baseline.get(k), feature.get(k)
        delta = ""
        if isinstance(b, (int, float)) and isinstance(f, (int, float)):
            delta = _fmt(f - b)
        print(f"{k:<18} {_fmt(b):>12} {_fmt(f):>12} {delta:>12}")


if __name__ == "__main__":
    main(sys.argv)
