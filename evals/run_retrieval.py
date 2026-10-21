"""Retrieval ablation: score every retrieval mode against the labelled question set.

    python evals/run_retrieval.py                       # all modes, writes evals/results/
    python evals/run_retrieval.py --check evals/baseline.json   # CI regression gate

No API key needed: this measures whether the right GOV.UK guide part is
retrieved, independently of any LLM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from govrag.chunk import load_chunks
from govrag.metrics import doc_ranking, hit_at_k, ndcg_at_k, reciprocal_rank
from govrag.retrieve import MODES, Retriever

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_questions(path: Path = ROOT / "questions.jsonl") -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def evaluate(retriever: Retriever, questions: list[dict], mode: str, pool: int = 10) -> tuple[dict, list[dict]]:
    rows, latencies = [], []
    for q in questions:
        gold = set(q["gold"])
        start = time.perf_counter()
        hits = retriever.search(q["question"], k=20, mode=mode, pool=pool)
        latencies.append((time.perf_counter() - start) * 1000)
        ranking = doc_ranking(hits)
        rows.append(
            {
                "id": q["id"],
                "hit@1": hit_at_k(ranking, gold, 1),
                "hit@3": hit_at_k(ranking, gold, 3),
                "hit@5": hit_at_k(ranking, gold, 5),
                "mrr": reciprocal_rank(ranking, gold),
                "ndcg@5": ndcg_at_k(ranking, gold, 5),
                "top3": ranking[:3],
                "tags": q["tags"],
            }
        )
    summary = {m: round(statistics.mean(r[m] for r in rows), 3) for m in ("hit@1", "hit@3", "hit@5", "mrr", "ndcg@5")}
    summary["p50_ms"] = round(statistics.median(latencies), 1)
    summary["p95_ms"] = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)
    return summary, rows


def by_tag(rows: list[dict], metric: str = "hit@3") -> dict[str, float]:
    tags = sorted({t for r in rows for t in r["tags"]})
    return {t: round(statistics.mean(r[metric] for r in rows if t in r["tags"]), 3) for t in tags}


def write_report(summaries: dict[str, dict], rows_by_mode: dict[str, list[dict]], questions: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(summaries, indent=2) + "\n")

    n = len(questions)
    lines = [
        "# Retrieval results",
        "",
        f"{n} answerable questions, scored at guide-part level. Latency is per query on an Apple M2 CPU.",
        "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in summaries.items():
        lines.append(
            f"| `{mode}` | {s['hit@1']:.3f} | {s['hit@3']:.3f} | {s['hit@5']:.3f} | {s['mrr']:.3f} | {s['ndcg@5']:.3f} | {s['p50_ms']} | {s['p95_ms']} |"
        )

    tag_table = {mode: by_tag(rows) for mode, rows in rows_by_mode.items()}
    tags = sorted({t for scores in tag_table.values() for t in scores})
    lines += ["", "## Hit@3 by question type", "", "| Mode | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for mode, scores in tag_table.items():
        lines.append(f"| `{mode}` | " + " | ".join(f"{scores[t]:.2f}" for t in tags) + " |")

    best = max(summaries, key=lambda m: summaries[m]["mrr"])
    by_id = {q["id"]: q for q in questions}
    misses = [r for r in rows_by_mode[best] if r["hit@3"] == 0]
    lines += ["", f"## Misses for `{best}` (gold not in top 3)", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        q = by_id[r["id"]]
        lines.append(
            f"- **{r['id']}** {q['question']}  \n  gold: `{', '.join(q['gold'])}` got: `{', '.join(r['top3'])}`"
        )
    (RESULTS / "retrieval.md").write_text("\n".join(lines) + "\n")


def check(summaries: dict[str, dict], baseline_path: Path, tolerance: float) -> int:
    baseline = json.loads(baseline_path.read_text())
    failures = []
    for mode, metrics in baseline.items():
        if mode not in summaries:
            continue
        for metric in ("hit@3", "mrr"):
            if summaries[mode][metric] < metrics[metric] - tolerance:
                failures.append(f"{mode} {metric}: {summaries[mode][metric]} < baseline {metrics[metric]}")
    for failure in failures:
        print("REGRESSION", failure)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--check", type=Path, help="fail if hit@3 or MRR drops below this baseline")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument(
        "--pool-sweep", nargs="*", type=int, metavar="N", help="also score rerank modes at these candidate pool sizes"
    )
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["answerable"]]
    retriever = Retriever.build(load_chunks())
    retriever.search("warm up", mode="hybrid_rerank")  # load models before timing

    summaries, rows_by_mode = {}, {}
    for mode in args.modes:
        summaries[mode], rows_by_mode[mode] = evaluate(retriever, questions, mode)
        print(mode, summaries[mode])

    write_report(summaries, rows_by_mode, questions)
    if args.pool_sweep:
        lines = [
            "# Reranker candidate pool sweep",
            "",
            "| Mode | Pool | Hit@1 | Hit@3 | MRR | p50 ms |",
            "|---|---|---|---|---|---|",
        ]
        for mode in ("dense_rerank", "hybrid_rerank"):
            for pool in args.pool_sweep:
                s, _ = evaluate(retriever, questions, mode, pool)
                lines.append(
                    f"| `{mode}` | {pool} | {s['hit@1']:.3f} | {s['hit@3']:.3f} | {s['mrr']:.3f} | {s['p50_ms']} |"
                )
                print(mode, pool, s)
        (RESULTS / "pool_sweep.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS / 'retrieval.md'}")
    if args.check:
        sys.exit(check(summaries, args.check, args.tolerance))


if __name__ == "__main__":
    main()
