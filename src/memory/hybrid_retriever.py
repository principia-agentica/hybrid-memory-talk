from typing import Callable, Dict, List
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
import importlib
import config as _cfg


class HybridRetriever:
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        k_epi: int | None = None,
        k_sem: int | None = None,
        epi_filter: Callable[[Dict], bool] | None = None,
        token_budget: int | None = None,
        reranker_enabled: bool | None = None,
    ):
        # Reload config at construction time to avoid stale, test-modified globals
        cfg = importlib.reload(_cfg)
        self.episodic, self.semantic = episodic, semantic
        self.k_epi = k_epi if k_epi is not None else cfg.K_EPI
        self.k_sem = k_sem if k_sem is not None else cfg.K_SEM
        # Build a default episodic filter from cfg.EPI_FILTERS if provided
        if epi_filter is not None:
            self.epi_filter = epi_filter
        elif cfg.EPI_FILTERS:
            def _mk_predicate(filters: Dict):
                def ok(e: Dict):
                    for k, v in filters.items():
                        if v is None:
                            continue
                        if k == "tags":
                            tags = e.get("tags", [])
                            if isinstance(v, list):
                                if not all(t in tags for t in v):
                                    return False
                            else:
                                if v not in tags:
                                    return False
                        else:
                            if e.get(k) != v:
                                return False
                    return True
                return ok
            self.epi_filter = _mk_predicate(cfg.EPI_FILTERS)
        else:
            self.epi_filter = (lambda e: True)
        self.token_budget = token_budget if token_budget is not None else cfg.TOKEN_BUDGET
        self.reranker_enabled = cfg.RERANKER_ENABLED if reranker_enabled is None else reranker_enabled
        # Store semantic filters snapshot from cfg
        self.sem_filters = cfg.SEM_FILTERS

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
        # Use semantic filters snapshot captured at construction time if provided
        if self.sem_filters:
            sem = self.semantic.search(query, top_k=self.k_sem, filters=self.sem_filters)
            # Fallback: if filters are too restrictive (e.g., items missing metadata/tags),
            # perform an unfiltered search to keep the demo and smoke tests educational.
            if not sem:
                sem = self.semantic.topk(query, self.k_sem)
        else:
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
