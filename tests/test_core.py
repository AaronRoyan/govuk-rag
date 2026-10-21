from types import SimpleNamespace

from govrag.chunk import Chunk, chunk_doc
from govrag.generate import ABSTAIN, answer
from govrag.metrics import contains_all, doc_ranking, hit_at_k, ndcg_at_k, reciprocal_rank
from govrag.retrieve import Hit, rrf, tokenize

DOC = {
    "doc_id": "guide/part",
    "title": "Guide",
    "part_title": "Part",
    "url": "https://www.gov.uk/guide/part",
    "text": "Intro line.\n## First section\n" + "\n".join(f"Sentence number {i} about notice." for i in range(40)),
}


def make_hit(doc_id: str, text: str = "text") -> Hit:
    chunk = Chunk(f"{doc_id}#0", doc_id, "Guide", "Part", "", f"https://www.gov.uk/{doc_id}", text)
    return Hit(chunk, 1.0)


def test_chunks_respect_size_and_keep_section_headers():
    chunks = chunk_doc(DOC, max_words=40)
    assert len(chunks) > 2
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert chunks[0].section == "" and chunks[1].section == "First section"
    assert chunks[1].index_text.startswith("Guide > Part > First section\n")


def test_consecutive_chunks_overlap_by_one_line():
    a, b = chunk_doc(DOC, max_words=40)[1:3]
    assert a.text.splitlines()[-1] == b.text.splitlines()[0]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("How much is the SSP, per week?") == ["much", "ssp", "per", "week"]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert [idx for idx, _ in fused][:2] == [1, 3]


def test_ranking_metrics():
    ranking = doc_ranking([make_hit("a"), make_hit("a"), make_hit("b"), make_hit("c")])
    assert ranking == ["a", "b", "c"]
    assert hit_at_k(ranking, {"b"}, 1) == 0 and hit_at_k(ranking, {"b"}, 2) == 1
    assert reciprocal_rank(ranking, {"c"}) == 1 / 3
    assert ndcg_at_k(ranking, {"a"}, 5) == 1.0


def test_fact_matching_is_tolerant_of_formatting():
    assert contains_all("You get a four-week trial period.", ["4 week"])
    assert contains_all("Up to 28 Weeks of SSP at £123.25", ["28 weeks", "123.25"])
    assert contains_all("one 20-min break", ["20 minute"])
    assert not contains_all("You get 2 weeks", ["12 weeks"])
    assert not contains_all("You get 12 weeks", ["2 weeks"])


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def fake_response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def test_answer_maps_citations_back_to_source_urls():
    hits = [make_hit("a/x"), make_hit("b/y")]
    blocks = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="Yes. ", citations=None),
        SimpleNamespace(
            type="text", text="You get 28 days.", citations=[SimpleNamespace(document_index=1, cited_text="28 days")]
        ),
    ]
    client = FakeClient(fake_response(blocks))
    result = answer("How much holiday?", hits, client=client)

    assert result.text == "Yes. You get 28 days."
    assert result.cited_urls == ["https://www.gov.uk/b/y"]
    assert not result.abstained
    sent = client.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in sent] == ["document", "document", "text"]
    assert all(b["citations"] == {"enabled": True} for b in sent[:2])


def test_answer_detects_abstention_and_refusal():
    hits = [make_hit("a/x")]
    abstain = answer(
        "?", hits, client=FakeClient(fake_response([SimpleNamespace(type="text", text=ABSTAIN, citations=None)]))
    )
    assert abstain.abstained and not abstain.refused

    refused = answer("?", hits, client=FakeClient(fake_response([], stop_reason="refusal")))
    assert refused.abstained and refused.refused


def test_answer_without_hits_skips_the_api():
    assert answer("anything", [], client=None).abstained
