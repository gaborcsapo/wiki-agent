"""Dev-only: summarize the latest FRAMES eval log for hill-climbing.

Prints per-sample correctness + retrieval grounding, and for each failing
sample shows the gold vs. read Wikipedia pages (which were missed). Helps
investigate failures one-by-one between tuning cycles.

Usage (from eval/):
    uv run python scripts/analyze_run.py            # latest log
    uv run python scripts/analyze_run.py <log.eval> # a specific log
"""

from __future__ import annotations

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

from wiki_eval.scorers import _fetched_pages, _normalize_wiki_url

LOGS = Path(__file__).resolve().parent.parent / "logs"


def _latest_log() -> Path:
    logs = sorted(LOGS.glob("*.eval"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit("no .eval logs found")
    return logs[-1]


def _corr(sample) -> str:
    # The correctness_judge() scorer is built on model_graded_qa, so Inspect
    # registers the score under that name.
    score = sample.scores.get("model_graded_qa") or sample.scores.get("correctness_judge")
    return str(getattr(score, "value", "?"))


def _grounding(sample):
    score = sample.scores.get("retrieval_grounding")
    if score is None:
        return None, {}
    return score.value, (score.metadata or {})


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_log()
    log = read_eval_log(str(path))
    samples = log.samples or []

    n = len(samples)
    n_correct = 0
    n_capped = 0
    recalls = []

    print(f"log: {path.name}   samples: {n}")
    print("=" * 100)

    for i, sample in enumerate(samples):
        corr = _corr(sample)
        is_correct = corr.upper().startswith("C")
        n_correct += int(is_correct)
        gvalue, gmeta = _grounding(sample)
        recall = (gvalue or {}).get("recall", 0.0) if isinstance(gvalue, dict) else 0.0
        recalls.append(recall)

        q = (sample.input or "")[:90].replace("\n", " ")
        target = str(sample.target)[:60].replace("\n", " ")
        full_answer = (sample.output.completion or "") if sample.output else ""
        steps_used = (sample.metadata or {}).get("steps", 0)
        if steps_used and steps_used >= 20:  # hit the FRAMES step budget
            n_capped += 1
        answer = full_answer[:80].replace("\n", " ")

        flag = "OK " if is_correct else "XX "
        print(
            f"[{i:02d}] {flag} corr={corr:<12} "
            f"recall={recall:.2f} ({gmeta.get('n_hit', '?')}/{gmeta.get('n_gold', '?')}) "
            f"read={gmeta.get('n_read', '?')}"
        )
        print(f"     Q: {q}")
        print(f"     target: {target}")
        print(f"     answer: {answer}")

        if not is_correct:
            gold = {
                slug
                for url in (sample.metadata or {}).get("reference_pages", [])
                if (slug := _normalize_wiki_url(url))
            }
            steps = (sample.metadata or {}).get("trajectory", {}).get("steps", [])
            read = _fetched_pages(steps)
            missed = sorted(gold - read)
            print(f"     gold:   {sorted(gold)}")
            print(f"     read:   {sorted(read)}")
            print(f"     missed: {missed}")
        print("-" * 100)

    acc = n_correct / n if n else 0.0
    mean_recall = sum(recalls) / n if n else 0.0
    print(
        f"SUMMARY  correctness={acc:.3f} ({n_correct}/{n})   "
        f"mean_recall={mean_recall:.3f}   step_capped={n_capped}/{n}"
    )


if __name__ == "__main__":
    main()
