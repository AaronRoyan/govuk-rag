"""Grounded answer generation with Claude and the Citations API.

Retrieved chunks are passed as citable `document` blocks, so every claim in
the answer comes back with a pointer to the exact passage it was drawn from.
Citations are parsed by the API, not regex-matched out of model text, so a
citation can never point at a document that was not provided.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic

from govrag.retrieve import Hit

MODEL = os.environ.get("GOVRAG_MODEL", "claude-opus-5-5")
ABSTAIN = "I couldn't find this in the GOV.UK employment guidance I have."

SYSTEM = f"""You answer questions from members of the public about UK employment rights, using only the GOV.UK guidance documents provided with each question.

Use only what the documents say. If you know something relevant that the documents don't state, leave it out: the point of this service is that every statement can be traced to official guidance.

Answer in plain English in 2 to 5 sentences, leading with the direct answer (for example "Yes" or the figure asked for). Mention a condition or exception only when it changes the answer for the person asking.

If the documents don't answer the question, reply with exactly this sentence and nothing else: {ABSTAIN}"""


@dataclass(frozen=True)
class Citation:
    url: str
    title: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    refused: bool = False
    model: str = MODEL
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cited_urls(self) -> list[str]:
        return list(dict.fromkeys(c.url for c in self.citations))


def _documents(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
            "title": hit.chunk.header,
            "context": f"Source: {hit.chunk.url}",
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def answer(question: str, hits: list[Hit], client: anthropic.Anthropic | None = None, effort: str = "low") -> Answer:
    if not hits:
        return Answer(text=ABSTAIN, abstained=True)

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": effort},
        # On a safety-classifier decline, retry server-side on Anthropic's
        # recommended fallback model instead of returning the refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": [*_documents(hits), {"type": "text", "text": question}]}],
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

    if response.stop_reason == "refusal":
        return Answer(text=ABSTAIN, abstained=True, refused=True, model=response.model, **usage)

    parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking and fallback blocks carry no answer text
        parts.append(block.text)
        for c in block.citations or []:
            chunk = hits[c.document_index].chunk
            citations.append(Citation(url=chunk.url, title=chunk.header, cited_text=c.cited_text))

    text = "".join(parts).strip()
    return Answer(
        text=text,
        citations=citations,
        abstained=ABSTAIN.rstrip(".") in text,
        model=response.model,
        **usage,
    )
