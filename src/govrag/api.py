"""HTTP API: `GET /search` works offline, `POST /ask` needs Claude credentials.

uvicorn govrag.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from govrag.chunk import load_chunks
from govrag.generate import answer
from govrag.retrieve import MODES, Mode, Retriever

state: dict[str, Retriever] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["retriever"] = Retriever.build(load_chunks())
    state["retriever"].search("warm up")
    yield
    state.clear()


app = FastAPI(title="GOV.UK employment rights Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"ok": "retriever" in state}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(5, ge=1, le=20), mode: Mode = "hybrid_rerank"):
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    hits = state["retriever"].search(q, k=k, mode=mode)
    return [
        {"score": h.score, "doc_id": h.chunk.doc_id, "title": h.chunk.header, "url": h.chunk.url, "text": h.chunk.text}
        for h in hits
    ]


@app.post("/ask")
def ask(req: AskRequest):
    hits = state["retriever"].search(req.question, k=req.k)
    result = answer(req.question, hits)
    return {
        "answer": result.text,
        "abstained": result.abstained,
        "citations": [asdict(c) for c in result.citations],
        "model": result.model,
    }


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Employment rights Q&amp;A</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;color:#0b0c0c}
input{width:100%;padding:.6rem;font:inherit;border:2px solid #0b0c0c}button{margin-top:.5rem;padding:.5rem 1rem;font:inherit;background:#00703c;color:#fff;border:0}
.src{font-size:.9rem;border-left:4px solid #b1b4b6;padding-left:.75rem;margin:.75rem 0}.src a{color:#1d70b8}</style>
<h1>Ask about your employment rights</h1>
<p>Answers come only from GOV.UK guidance, with a source for every claim. This is not legal advice.</p>
<form id="f"><input id="q" placeholder="e.g. How much notice do I need to give when I resign?" autofocus><button>Ask</button></form>
<div id="out"></div>
<script>
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.getElementById("f").onsubmit = async e => {
  e.preventDefault(); const out = document.getElementById("out"); out.textContent = "Thinking…";
  const r = await fetch("/ask", {method:"POST", headers:{"content-type":"application/json"}, body: JSON.stringify({question: document.getElementById("q").value})});
  if (!r.ok) { out.textContent = "Something went wrong (" + r.status + ")."; return; }
  const d = await r.json(); const seen = new Set();
  out.innerHTML = "<p>" + esc(d.answer) + "</p>" + d.citations.filter(c => !seen.has(c.cited_text) && seen.add(c.cited_text))
    .map(c => `<div class="src">“${esc(c.cited_text)}”<br><a href="${esc(c.url)}">${esc(c.title)}</a></div>`).join("");
};
</script></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
