"""Split GOV.UK guide parts into retrievable chunks.

Chunks never cross a section heading, and each one carries a breadcrumb
header ("Guide > Part > Section") that is prepended before indexing, so a
chunk like "You must give at least 1 week's notice" still knows which guide
it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    part_title: str
    section: str
    url: str
    text: str

    @property
    def header(self) -> str:
        crumbs = [self.title]
        if self.part_title and self.part_title != self.title:
            crumbs.append(self.part_title)
        if self.section:
            crumbs.append(self.section)
        return " > ".join(crumbs)

    @property
    def index_text(self) -> str:
        return f"{self.header}\n{self.text}"


def _sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    return [(heading, lines) for heading, lines in sections if lines]


def _pack(lines: list[str], max_words: int) -> list[list[str]]:
    """Greedily pack lines into windows of at most max_words, repeating the
    last line of each window at the start of the next for continuity."""
    windows: list[list[str]] = []
    current: list[str] = []
    count = 0
    for line in lines:
        n = len(line.split())
        if current and count + n > max_words:
            windows.append(current)
            carry = current[-1]
            current, count = ([carry], len(carry.split())) if len(carry.split()) < max_words // 3 else ([], 0)
        current.append(line)
        count += n
    if current:
        windows.append(current)
    return windows


def chunk_doc(doc: dict, max_words: int = 160) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, lines in _sections(doc["text"]):
        for window in _pack(lines, max_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}#{len(chunks)}",
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    part_title=doc["part_title"],
                    section=section,
                    url=doc["url"],
                    text="\n".join(window),
                )
            )
    return chunks


def load_chunks(corpus: Path = Path("data/corpus.jsonl"), max_words: int = 160) -> list[Chunk]:
    with corpus.open() as f:
        docs = [json.loads(line) for line in f]
    return [chunk for doc in docs for chunk in chunk_doc(doc, max_words)]
