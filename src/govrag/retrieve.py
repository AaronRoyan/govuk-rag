"""Retrieval: BM25, dense, hybrid (reciprocal rank fusion) and cross-encoder reranking.

All four modes share one interface so the eval harness can ablate them:

    retriever = Retriever.build(load_chunks())
    hits = retriever.search("how much notice do I give?", mode="hybrid_rerank")
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from govrag.chunk import Chunk

Mode = Literal["bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# bge v1.5 retrieves better when queries (not passages) carry this instruction.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or "
    "the to what when where which who will with you your".split()
)
CACHE_DIR = Path(".cache")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder, reranker=None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self._reranker = reranker
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embed_model: str = EMBED_MODEL) -> Retriever:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(embed_model)
        texts = [c.index_text for c in chunks]
        digest = hashlib.sha1((embed_model + "\x00".join(texts)).encode()).hexdigest()[:12]
        cache = CACHE_DIR / f"emb-{digest}.npy"
        if cache.exists():
            embeddings = np.load(cache)
        else:
            embeddings = np.array(list(embedder.embed(texts)), dtype=np.float32)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(cache, embeddings)
        return cls(chunks, embeddings, embedder)

    @property
    def reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        return self._reranker

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:n]]

    def _dense_scores(self, query: str) -> np.ndarray:
        q = next(iter(self.embedder.embed([QUERY_INSTRUCTION + query])))
        q = q / np.linalg.norm(q)
        return self.embeddings @ q

    def _dense_ranking(self, query: str, n: int) -> list[int]:
        return [int(i) for i in np.argsort(-self._dense_scores(query))[:n]]

    def _rerank(self, query: str, candidates: list[int], k: int) -> list[Hit]:
        docs = [self.chunks[i].index_text for i in candidates]
        scores = np.array(list(self.reranker.rerank(query, docs)))
        return [Hit(self.chunks[candidates[j]], float(scores[j])) for j in np.argsort(-scores)[:k]]

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid_rerank", pool: int = 10) -> list[Hit]:
        """Return the top k chunks. Rerank modes rescore the first `pool` candidates."""
        if mode == "bm25":
            scores = self.bm25.get_scores(tokenize(query))
            return [Hit(self.chunks[i], float(scores[i])) for i in self._bm25_ranking(query, k)]
        if mode == "dense":
            scores = self._dense_scores(query)
            return [Hit(self.chunks[int(i)], float(scores[i])) for i in np.argsort(-scores)[:k]]
        if mode == "dense_rerank":
            return self._rerank(query, self._dense_ranking(query, pool), k)

        fused = rrf([self._bm25_ranking(query, pool), self._dense_ranking(query, pool)])
        if mode == "hybrid":
            return [Hit(self.chunks[i], s) for i, s in fused[:k]]
        return self._rerank(query, [i for i, _ in fused[:pool]], k)
