"""Per-run breakdown for the multilingual_qa benchmark + failure listing.

    uv run python -m wiki_eval.analyze_multilingual <log.eval>

Prints overall accuracy, per-category and per-language accuracy, and a list of
failures (question + gold + the `lang` codes the agent actually queried) to drive
the next improvement cycle. The pure aggregator `_summarize` is unit-tested.
"""

from __future__ import annotations

import sys
from collections import defaultdict

CORRECT = "C"
SCORE_KEYS = ("multilingual_correctness", "model_graded_qa")


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    out = {"n": n, "accuracy": 0.0, "by_category": {}, "by_language": {}}
    if not n:
        return out
    out["accuracy"] = sum(1 for r in records if r["correct"]) / n
    for field in ("category", "language"):
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in records:
            groups[r.get(field, "?")].append(r["correct"])
        out[f"by_{field}"] = {k: sum(v) / len(v) for k, v in sorted(groups.items())}
    return out


def _record(sample) -> dict:
    scores = getattr(sample, "scores", None) or {}
    sc = next((scores[k] for k in SCORE_KEYS if k in scores), None)
    correct = bool(sc is not None and getattr(sc, "value", None) == CORRECT)
    meta = getattr(sample, "metadata", None) or {}
    steps = (meta.get("trajectory") or {}).get("steps", [])
    # Record "?" when a tool call omitted `lang` (English by default) so the
    # failure listing distinguishes a deliberate en query from an unset one.
    langs = sorted({(s.get("tool_input") or {}).get("lang", "?")
                    for s in steps if s.get("kind") == "tool_call"})
    return {
        "correct": correct,
        "category": meta.get("category", "?"),
        "language": meta.get("language", "?"),
        "question": getattr(sample, "input", ""),
        "target": getattr(sample, "target", ""),
        "langs_used": langs,
    }


def _fmt_pct(d: dict) -> str:
    return ", ".join(f"{k}={v:.0%}" for k, v in d.items())


def main(argv: list[str]) -> None:
    from inspect_ai.log import read_eval_log
    log = read_eval_log(argv[1])
    records = [_record(s) for s in (log.samples or [])]
    s = _summarize(records)
    print(f"n={s['n']}  accuracy={s['accuracy']:.3f}")
    print("by category:", _fmt_pct(s["by_category"]))
    print("by language:", _fmt_pct(s["by_language"]))
    print("\nFAILURES:")
    for r in records:
        if not r["correct"]:
            q = r["question"][:90].replace("\n", " ")
            print(f"  [{r['category']}/{r['language']}] langs={r['langs_used']}  {q}")
            print(f"      gold: {str(r['target'])[:90]}")


if __name__ == "__main__":
    main(sys.argv)
