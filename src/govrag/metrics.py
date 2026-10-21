"""Ranking metrics, computed at document (guide part) level.

Several chunks can come from the same part, so chunk rankings are collapsed
to their first occurrence per document before scoring. A question can list
more than one gold document; any of them counts as relevant.
"""

from __future__ import annotations

import math
import re

from govrag.retrieve import Hit


def doc_ranking(hits: list[Hit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.doc_id not in seen:
            seen.append(hit.chunk.doc_id)
    return seen


def hit_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    return float(any(doc in gold for doc in ranking[:k]))


def reciprocal_rank(ranking: list[str], gold: set[str]) -> float:
    for rank, doc in enumerate(ranking, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc in enumerate(ranking[:k], start=1) if doc in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fourteen": "14",
    "twenty": "20",
}


def normalise(text: str) -> str:
    text = text.lower().replace("-", " ").replace("’", "'")
    text = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: _NUMBER_WORDS[m.group(1)], text)
    text = re.sub(r"(\d)\s+(?:minutes?|mins?)\b", r"\1 minute", text)
    text = re.sub(r"\b(\d+) (weeks?|days?|months?|hours?)'?", lambda m: f"{m.group(1)} {m.group(2).rstrip('s')}s", text)
    return re.sub(r"\s+", " ", text)


def contains_all(answer: str, required: list[str]) -> bool:
    """True if every required fact appears in the answer, ignoring case,
    hyphenation and spelled-out numbers ("four weeks" == "4-week")."""
    norm = normalise(answer)
    return all(re.search(rf"(?<!\w){re.escape(normalise(fact))}(?!\w)", norm) for fact in required)
