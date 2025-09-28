from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore


class HybridRetriever:
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        k_epi=3,
        k_sem=3,
        epi_filter=None,
    ):
        self.episodic, self.semantic = episodic, semantic
        self.k_epi, self.k_sem = k_epi, k_sem
        self.epi_filter = epi_filter or (lambda e: True)

    def retrieve(self, query):
        epi = self.episodic.topk(self.k_epi, where=self.epi_filter)
        sem = self.semantic.topk(query, self.k_sem)
        return epi + sem

    def __repr__(self):
        return f"HybridRetriever(episodic={self.episodic}, semantic={self.semantic}, k_epi={self.k_epi}, k_sem={self.k_sem})"
