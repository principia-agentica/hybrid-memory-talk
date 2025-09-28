import numpy as np


class SemanticStore:
    def __init__(self, encoder):
        self.encoder = encoder
        self.vecs, self.docs = [], []

    def add(self, doc):
        self.docs.append(doc)
        self.vecs.append(self.encoder.embed(doc))

    def topk(self, query, k=5):
        q = self.encoder.embed(query)
        sims = [
            float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8))
            for v in self.vecs
        ]
        idx = np.argsort(sims)[::-1][:k]
        return [self.docs[i] for i in idx]

    def __repr__(self):
        return f"SemanticStore(encoder={self.encoder}, num_docs={len(self.docs)})"
