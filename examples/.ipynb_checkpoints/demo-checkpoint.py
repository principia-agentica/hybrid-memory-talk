"""
Hybrid Memory Demo CLI

Run: python -m examples.demo

This script seeds a tiny semantic policy store, starts a short conversation
that triggers tool calls, performs hybrid retrieval (episodic + semantic),
and prints the retrieved context with provenance, the final answer from the
Agent's LLM stub, and where minimal traces were written.

It is intentionally small and offline-only to mirror the ideas from
context/091925-memory-in-agents.md and context/plan.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from agent import Agent
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.hybrid_retriever import HybridRetriever
from tracing.tracer import tracer as TRACER


# ---------------- Tiny offline encoder (hash sketch) ----------------
class TinyHashEncoder:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str):
        d = self.dim
        vec = [0.0] * d
        for tok in (text or "").lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            # Project into 4 positions with sign
            for i in range(4):
                idx = (h >> (i * 8)) % d
                sign = 1.0 if ((h >> (i * 2)) & 1) else -1.0
                vec[idx] += sign
        # Let the store normalize; return as list
        return vec


# ---------------- Policy seeding ----------------
POLICY_DIR = Path(__file__).parent / "policies"


def _read_policies() -> List[Dict]:
    items: List[Dict] = []
    for p in sorted(POLICY_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        section = lines[0].lstrip("# ")
        body = " ".join(lines[1:]) if len(lines) > 1 else section
        items.append(
            {
                "id": p.stem,
                "text": body,
                "metadata": {
                    "source": p.name,
                    "section": section,
                    "tags": ["policy"],
                    "pii": False,
                },
            }
        )
    return items


def seed_semantic_store(sem: SemanticStore) -> List[str]:
    ids = []
    for item in _read_policies():
        sem.upsert(item)
        ids.append(item["id"])
    return ids


# ---------------- Pretty printers ----------------

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_items(label: str, items: List[Dict]):
    print(f"\n{label} ({len(items)} items):")
    for it in items:
        kind = it.get("kind", "?")
        src = it.get("source", it.get("metadata", {}).get("source", "?"))
        text = it.get("text", "").strip()
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"- [{kind}] {src} :: {text}")


# ---------------- Simple metric ----------------

def recall_at_k(retriever: HybridRetriever, query: str, relevant_ids: List[str]) -> float:
    items = retriever.retrieve(query)
    got = {it.get("id") for it in items if it.get("kind") == "semantic"}
    rel = set(relevant_ids)
    if not rel:
        return 1.0
    return len(got & rel) / float(len(rel))


# ---------------- Main demo flow ----------------

def run_demo():
    print_header("Hybrid Memory Demo: Support Copilot")

    # Build stores and retriever (defaults from config.py)
    encoder = TinyHashEncoder()
    sem = SemanticStore(encoder=encoder)
    epi = EpisodicStore(max_len=50)

    # Seed policies
    seeded_ids = seed_semantic_store(sem)
    print("Seeded policy snippets:", ", ".join(seeded_ids))

    retriever = HybridRetriever(episodic=epi, semantic=sem)
    agent = Agent(retriever=retriever, task_id="support_demo", session="sess_1", tracer=TRACER)

    # User turn 1: triggers lookup + reset
    q1 = "Hi, I forgot my password. My email is ana@example.com. Can you reset it?"
    print_header("User Turn 1")
    print(q1)
    a1 = agent.answer(q1)

    # Show retrieved context after tools ran
    ctx_after = retriever.retrieve(q1)
    show_items("Retrieved context (after tools)", ctx_after)

    print_header("Assistant Answer 1")
    print(a1)

    # Follow-up turn (no new tools expected)
    q2 = "Thanks! What are the steps involved?"
    print_header("User Turn 2")
    print(q2)
    a2 = agent.answer(q2)

    ctx2 = retriever.retrieve(q2)
    show_items("Retrieved context", ctx2)

    print_header("Assistant Answer 2")
    print(a2)

    # Tiny metric: Recall@k for a canned query
    r = recall_at_k(retriever, "password reset steps", relevant_ids=["policy_password", "policy_steps"])
    print_header("Mini-metric")
    print(f"Recall@k (k≈defaults) for 'password reset steps': {r:.2f}")

    # Traces location
    print_header("Tracing")
    print("Wrote minimal traces to out/traces.jsonl (one row per span).")


if __name__ == "__main__":
    run_demo()
