"""Dev-only: build wiki_eval/datasets/frames.jsonl from google/frames-benchmark.

Not imported by the package and not covered by tests. Requires the `datasets`
dev dependency and network access.

Usage (from eval/):
    uv run python scripts/build_frames.py --limit 100
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "wiki_eval" / "datasets" / "frames.jsonl"


def _reference_pages(wiki_links) -> list[str]:
    """FRAMES stores reference URLs as a stringified Python list; parse to a list."""
    if isinstance(wiki_links, str):
        return list(ast.literal_eval(wiki_links))
    return list(wiki_links)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Rows to keep (0 = all 824).")
    args = parser.parse_args()

    dataset = load_dataset("google/frames-benchmark", split="test")
    rows = []
    for i, record in enumerate(dataset):
        if args.limit and i >= args.limit:
            break
        rows.append(
            {
                "input": record["Prompt"],
                "target": record["Answer"],
                "reference_pages": _reference_pages(record["wiki_links"]),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
