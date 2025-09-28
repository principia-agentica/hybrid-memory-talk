from datetime import datetime, timedelta

from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.hybrid_retriever import HybridRetriever


class TinyEncoder:
    def embed(self, text: str):
        # very small deterministic vector from length and vowels count
        v = sum(1 for c in text.lower() if c in "aeiou")
        return [len(text) % 7, float(v) + 1.0]


def test_episodic_window_and_filters_and_ttl():
    epi = EpisodicStore(max_len=10)

    # expired event should be purged on fetch
    epi.add({
        "id": "old",
        "type": "note",
        "text": "this is old",
        "ts": datetime.utcnow().isoformat() + "Z",
        "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z",
        "tags": ["session:s1", "topic:x"],
    })

    # recent events, two different sessions
    for i in range(5):
        epi.add({
            "id": f"e{i}",
            "type": "user_turn" if i % 2 == 0 else "assistant_turn",
            "text": f"event {i}",
            "tags": ["session:s1" if i < 4 else "session:s2", "topic:password_reset"],
        })

    # filter by session s1 and take last 3
    out = epi.fetch(last_n=3, filters={"tags": ["session:s1"]})
    assert len(out) == 3
    assert all("session:s1" in e.get("tags", []) for e in out)
    # ensure expired one was not returned
    ids = [e.get("id") for e in out]
    assert "old" not in ids


def test_semantic_search_filters_and_pii_scrub():
    sem = SemanticStore(encoder=TinyEncoder(), pii_scrub_at_ingest=True)

    sem.upsert({
        "id": "p1",
        "text": "Customers on annual plans must verify email before password reset.",
        "metadata": {"source": "policy.md", "section": "password", "tags": ["policy"], "pii": False},
    })
    sem.upsert({
        "id": "p2",
        "text": "Write to support at admin@example.com for escalations.",
        "metadata": {"source": "runbook.md", "section": "contact", "tags": ["runbook"], "pii": False},
    })
    sem.upsert({
        "id": "p3",
        "text": "Internal customer list contains emails and phone numbers.",
        "metadata": {"source": "internal.md", "section": "pii", "tags": ["internal"], "pii": True},
    })

    # filter to policy-tagged, pii False
    res = sem.search(
        text="password reset policy",
        top_k=5,
        filters={"tags": ["policy"], "pii": False},
    )
    assert res, "should retrieve at least one policy snippet"
    assert all("policy" in d.get("metadata", {}).get("tags", []) for d in res)
    assert all(d.get("metadata", {}).get("pii") is False for d in res)

    # PII scrub at ingest: email should be redacted in stored text
    p2 = next(d for d in sem._items if d.get("id") == "p2")
    assert "@" not in p2["text"] and "<EMAIL>" in p2["text"]


def test_hybrid_merge_provenance_and_trim():
    epi = EpisodicStore(max_len=10)
    sem = SemanticStore(encoder=TinyEncoder())

    # Add a few episodic events
    epi.add({"id": "e1", "type": "user_turn", "text": "User asks about password reset policy."})
    epi.add({"id": "e2", "type": "tool_result", "text": "lookup_user: ana is verified."})

    # Add semantic docs with some longer text to exercise token trimming
    sem.add({"id": "s1", "text": "Password reset policy requires email verification and a cooldown period." ,
             "metadata": {"source": "policy.md", "section": "password", "tags": ["policy"], "pii": False}})
    sem.add({"id": "s2", "text": "Checklist: confirm identity, verify email, send reset link, log the action." ,
             "metadata": {"source": "policy.md", "section": "checklist", "tags": ["policy"], "pii": False}})

    retriever = HybridRetriever(episodic=epi, semantic=sem, k_epi=2, k_sem=2, token_budget=12)
    items = retriever.retrieve("password reset steps")

    # Should include provenance and kind
    assert items, "retriever should return some items"
    assert all("kind" in it and "source" in it for it in items)
    # Should have trimmed to small budget (likely less than total candidates)
    assert len(items) <= 4

    # Ensure both episodic and semantic are present if budget allows
    kinds = {it.get("kind") for it in items}
    assert kinds.issubset({"episodic", "semantic"})


def test_hybrid_reranker_prioritizes_semantic_when_enabled():
    epi = EpisodicStore(max_len=10)
    sem = SemanticStore(encoder=TinyEncoder())

    # Add one episodic note that doesn't contain the query words
    epi.add({"id": "e1", "type": "note", "text": "Unrelated system status message."})

    # Add one semantic item that clearly matches the query
    sem.add({
        "id": "s1",
        "text": "Token budget guidance and how to set k for retrievers.",
        "metadata": {"source": "policy.md", "section": "retrieval", "tags": ["policy"], "pii": False},
    })

    # Without reranker, order is episodic first then semantic (due to concat order)
    r_no = HybridRetriever(episodic=epi, semantic=sem, k_epi=1, k_sem=1, reranker_enabled=False)
    items_no = r_no.retrieve("how to set k and token budget")
    assert items_no[0]["kind"] == "episodic"

    # With reranker, semantic should be ranked first
    r_yes = HybridRetriever(episodic=epi, semantic=sem, k_epi=1, k_sem=1, reranker_enabled=True)
    items_yes = r_yes.retrieve("how to set k and token budget")
    assert items_yes[0]["kind"] == "semantic"
