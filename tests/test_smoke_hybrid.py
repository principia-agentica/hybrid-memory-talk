from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.hybrid_retriever import HybridRetriever

class DummyEncoder:
    def embed(self, text: str):
        # vector unitario simple por longitud
        return [len(text) % 7, 1.0]

class DummyLLM:
    def generate(self, prompt: str):
        return f"[MOCK LLM] {prompt[:120]}..."

def test_smoke_hybrid():
    # epi = EpisodicStore(window_size=5, retention_by_category={"decision": 5, "note": 5})
    epi = EpisodicStore(max_len=5)
    sem = SemanticStore(encoder=DummyEncoder())

    epi.add(event={"id":"e1", "cat":"decision", "text":"Usar k_epi=2 y k_sem=2"})
    epi.add(event={"id":"e2", "cat":"note", "text":"Error 500 en /analyze"})
    sem.add({"id":"d1", "text":"Guía canónica de pipeline híbrido"})
    sem.add({"id":"d2", "text":"Checklists y límites de token"})

    retriever = HybridRetriever(
        episodic=epi, semantic=sem, k_epi=2, k_sem=2
    )
    items = retriever.retrieve(query="¿Cómo setear k y presupuesto de tokens?")
    prompt = (
        "Contexto:\n"
        + "\n".join(f"- {it['text']}" for it in items)
        + "\n\nPregunta: ¿Recomendación?"
    )
    llm = DummyLLM()
    output = llm.generate(prompt)
    print(output)

    # Smoke assertions for symmetry with the plan
    assert items, "retriever should return some items"
    # Ensure we got both episodic and semantic context
    kinds = [it.get("kind") for it in items]
    assert any(k == "episodic" for k in kinds), "should include episodic items"
    assert any(k == "semantic" for k in kinds), "should include semantic items"
    # Default order without reranker: episodic first
    assert items[0].get("kind") == "episodic"
    # Provenance/source should be present
    assert all("source" in it for it in items), "all items should carry provenance 'source'"
    # Should not exceed requested k's (subject to dedupe/trim)
    assert len(items) <= 4
