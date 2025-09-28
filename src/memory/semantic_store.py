import re
import numpy as np
from typing import Any, Dict, List, Optional


class SemanticStore:
    """
    Minimal in-memory vector store. Accepts an encoder with embed(text)->vector.
    Provides upsert/search with simple metadata filters and PII scrub at ingest.
    Backwards-compatible add() and topk() methods used by tests.
    """

    def __init__(self, encoder, pii_scrub_at_ingest: bool = True):
        self.encoder = encoder
        self._X: Optional[np.ndarray] = None  # matrix of vectors (n x d)
        self._items: List[Dict[str, Any]] = []
        self._id_to_row: Dict[str, int] = {}
        self.pii_scrub_at_ingest = pii_scrub_at_ingest

    # --- Helpers ---
    @staticmethod
    def _scrub_pii(text: str) -> str:
        # naive: remove emails
        return re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", text)

    def _ensure_matrix_row(self, vec: np.ndarray):
        vec = np.asarray(vec, dtype=float)
        nrm = np.linalg.norm(vec) or 1.0
        vec = vec / nrm
        if self._X is None:
            self._X = vec.reshape(1, -1)
        else:
            # pad dims if needed (in case encoder output size changes)
            d_current = self._X.shape[1]
            d_new = vec.shape[0]
            if d_new != d_current:
                # align by padding the smaller with zeros
                if d_new > d_current:
                    pad = np.zeros((self._X.shape[0], d_new - d_current))
                    self._X = np.hstack([self._X, pad])
                elif d_current > d_new:
                    vec = np.pad(vec, (0, d_current - d_new))
            self._X = np.vstack([self._X, vec])

    # --- New API per plan ---
    def upsert(self, item: Dict[str, Any]) -> None:
        """Insert or replace by id. Computes embedding from text."""
        _id = item.get("id")
        text = item.get("text", "")
        if not text:
            raise ValueError("item.text required")
        if self.pii_scrub_at_ingest:
            text = self._scrub_pii(text)
        emb = self.encoder.embed(text)
        if _id and _id in self._id_to_row:
            row = self._id_to_row[_id]
            self._items[row] = {**item, "text": text}
            # replace vector
            vec = np.asarray(emb, dtype=float)
            nrm = np.linalg.norm(vec) or 1.0
            vec = vec / nrm
            d_current = self._X.shape[1]
            if vec.shape[0] != d_current:
                if vec.shape[0] < d_current:
                    vec = np.pad(vec, (0, d_current - vec.shape[0]))
                else:
                    pad = np.zeros((self._X.shape[0], vec.shape[0] - d_current))
                    self._X = np.hstack([self._X, pad])
            self._X[row] = vec
        else:
            self._items.append({**item, "text": text})
            self._id_to_row[_id or str(len(self._items)-1)] = len(self._items) - 1
            self._ensure_matrix_row(np.asarray(emb, dtype=float))

    def search(self, text: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        if not self._items:
            return []
        q = self.encoder.embed(text)
        q = np.asarray(q, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        # filter pre-ranking
        idxs = list(range(len(self._items)))
        if filters:
            def ok(meta):
                for k, v in filters.items():
                    if k == "tags":
                        tags = meta.get("metadata", {}).get("tags", [])
                        if isinstance(v, list):
                            if not all(t in tags for t in v):
                                return False
                        else:
                            if v not in tags:
                                return False
                    else:
                        if meta.get("metadata", {}).get(k) != v:
                            return False
                return True
            idxs = [i for i in idxs if ok(self._items[i])]
            if not idxs:
                return []
        X = self._X[idxs]
        sims = X @ q
        order = np.argsort(-sims)[:top_k]
        return [self._items[idxs[i]] for i in order]

    # --- Backwards-compatible API used by tests ---
    def add(self, doc: Dict[str, Any]):
        # allow tests to pass a plain dict with text
        _id = doc.get("id")
        self.upsert({"id": _id, "text": doc.get("text", ""), "metadata": doc.get("metadata", {})})

    def topk(self, query: str, k: int = 5):
        return self.search(query, top_k=k)

    def __repr__(self):
        return f"SemanticStore(encoder={self.encoder}, num_docs={len(self._items)})"
