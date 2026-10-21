"""End-to-end answer evals (needs ANTHROPIC_API_KEY or an `ant auth login` profile).

For every question: retrieve with the production mode, generate a cited
answer, then score it four ways:

- abstention: answers out-of-scope questions with the fixed "couldn't find"
  sentence, and does not abstain on answerable ones
- key facts: deterministic check that required figures (e.g. "£123.25") appear
- citation precision: share of cited GOV.UK parts that are gold for the question
- judge: an LLM grader compares the answer to the reference answer and to the
  retrieved passages, flagging wrong answers and claims the passages don't support

A full run is 80 answer calls plus 70 judge calls, roughly US$3 on claude-opus-5-5.

    python evals/run_generation.py --limit 10     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

from govrag.chunk import load_chunks
from govrag.generate import Answer, answer
from govrag.metrics import contains_all
from govrag.retrieve import Retriever

ROOT = Path(__file__).resolve().parent
JUDGE_MODEL = os.environ.get("GOVRAG_JUDGE_MODEL", "claude-opus-5-5")

JUDGE_PROMPT = """You are grading an answer from a question-answering service built on GOV.UK employment guidance.

<question>{question}</question>

<reference_answer>{reference}</reference_answer>

<retrieved_passages>
{passages}
</retrieved_passages>

<answer_to_grade>{answer}</answer_to_grade>

Grade correctness against the reference answer: "correct" if it gives the same substantive answer (wording can differ and extra accurate detail is fine), "partial" if it is right but misses something the reference treats as essential, "incorrect" if it contradicts the reference or answers a different question.

Separately, list any factual claims in the answer that the retrieved passages do not support. Claims that are true but absent from the passages still count as unsupported."""


class Grade(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    unsupported_claims: list[str]
    reason: str


def judge(client: anthropic.Anthropic, q: dict, ans: Answer, passages: str) -> Grade:
    response = client.beta.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        output_format=Grade,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=q["question"], reference=q["reference"], passages=passages, answer=ans.text
                ),
            }
        ],
    )
    return response.parsed_output


def run_one(client: anthropic.Anthropic, retriever: Retriever, q: dict) -> dict:
    hits = retriever.search(q["question"], k=5)
    ans = answer(q["question"], hits, client=client)
    gold = set(q["gold"])
    cited_docs = [url.removeprefix("https://www.gov.uk/") for url in ans.cited_urls]

    row = {
        "id": q["id"],
        "answerable": q["answerable"],
        "answer": ans.text,
        "abstained": ans.abstained,
        "refused": ans.refused,
        "cited": cited_docs,
        "abstention_correct": ans.abstained != q["answerable"],
        "input_tokens": ans.input_tokens,
        "output_tokens": ans.output_tokens,
    }
    if q["answerable"] and not ans.abstained:
        row["facts_ok"] = contains_all(ans.text, q["must_include"]) if q["must_include"] else None
        row["citation_precision"] = (sum(d in gold for d in cited_docs) / len(cited_docs)) if cited_docs else 0.0
        passages = "\n\n".join(f"[{h.chunk.header}]\n{h.chunk.text}" for h in hits)
        row["grade"] = judge(client, q, ans, passages).model_dump()
    return row


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else float("nan")


def summarise(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["answerable"] and not r["abstained"]]
    graded = [r["grade"] for r in answered]
    return {
        "questions": len(rows),
        "abstention_accuracy": mean([r["abstention_correct"] for r in rows]),
        "false_abstentions": sum(r["answerable"] and r["abstained"] for r in rows),
        "missed_abstentions": sum(not r["answerable"] and not r["abstained"] for r in rows),
        "key_fact_recall": mean([r["facts_ok"] for r in answered if r["facts_ok"] is not None]),
        "citation_precision": mean([r["citation_precision"] for r in answered]),
        "answers_with_citations": mean([bool(r["cited"]) for r in answered]),
        "correct": mean([g["correctness"] == "correct" for g in graded]),
        "correct_or_partial": mean([g["correctness"] != "incorrect" for g in graded]),
        "fully_supported": mean([not g["unsupported_claims"] for g in graded]),
        "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows)),
        "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only run the first N answerable and N unanswerable questions")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with (ROOT / "questions.jsonl").open() as f:
        questions = [json.loads(line) for line in f]
    if args.limit:
        questions = [q for q in questions if q["answerable"]][: args.limit] + [
            q for q in questions if not q["answerable"]
        ][: args.limit]

    client = anthropic.Anthropic()
    retriever = Retriever.build(load_chunks())
    retriever.search("warm up")  # load the reranker once, before threads share it

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda q: run_one(client, retriever, q), questions))

    summary = summarise(rows)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "generation.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (results / "generation_answers.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
