from typing import Callable, Dict, List
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore


class HybridRetriever:
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        k_epi: int = 3,
        k_sem: int = 3,
        epi_filter: Callable[[Dict], bool] | None = None,
        token_budget: int = 1600,
        reranker_enabled: bool = False,
    ):
        self.episodic, self.semantic = episodic, semantic
        self.k_epi, self.k_sem = k_epi, k_sem
        self.epi_filter = epi_filter or (lambda e: True)
        self.token_budget = token_budget
        self.reranker_enabled = reranker_enabled

    def _count_tokens(self, text: str) -> int:
        # naive estimate: 1.3x words
        return int(len((text or "").split()) * 1.3) or 1

    def _dedupe(self, items: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for it in items:
            key = (it.get("source"), hash(it.get("text")))
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    def _annotate_provenance(self, epi: List[Dict], sem: List[Dict]) -> List[Dict]:
        out = []
        for e in epi:
            prov = e.get("provenance") or f"episodic@{e.get('ts','#')}#{e.get('type', e.get('cat','event'))}"
            out.append({**e, "kind": "episodic", "source": prov})
        for s in sem:
            md = s.get("metadata", {})
            prov = s.get("provenance") or f"{md.get('source','doc')}#{md.get('section', md.get('id','section'))}"
            out.append({**s, "kind": "semantic", "source": prov})
        return out

    def _trim(self, items: List[Dict]) -> List[Dict]:
        used = 0
        out = []
        for it in items:
            t = self._count_tokens(it.get("text", ""))
            if used + t <= self.token_budget:
                out.append(it)
                used += t
            else:
                break
        return out

    def retrieve(self, query: str) -> List[Dict]:
        epi = self.episodic.topk(self.k_epi, where=self.epi_filter)
        sem = self.semantic.topk(query, self.k_sem)
        items = self._annotate_provenance(epi, sem)
        # optional very light rerank: prefer semantic that mention query words
        if self.reranker_enabled:
            q_words = set(query.lower().split())
            def score(it):
                s = 0
                if it.get("kind") == "semantic":
                    s += 1
                t_words = set((it.get("text") or "").lower().split())
                s += len(q_words & t_words) * 0.1
                if it.get("kind") == "episodic":
                    s += 0.05  # slight recency bias
                return s
            items = sorted(items, key=score, reverse=True)
        # dedupe and trim
        items = self._dedupe(items)
        items = self._trim(items)
        return items

    def __repr__(self):
        return (
            f"HybridRetriever(episodic={self.episodic}, semantic={self.semantic}, "
            f"k_epi={self.k_epi}, k_sem={self.k_sem})"
        )
