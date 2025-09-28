from memory.hybrid_retriever import HybridRetriever


class Agent:
    def __init__(self, retriever: HybridRetriever, llm):
        self.retriever, self.llm = retriever, llm

    def note_event(self, event):
        # tool: guardar evento
        self.retriever.episodic.add(event)
        return {"ok": True}

    def answer(self, query: str):
        ctx = self.retriever.retrieve(query)
        prompt = f"Contexto:\n{ctx}\n\nPregunta: {query}\nRespuesta:"
        return self.llm(prompt)

    def __repr__(self):
        return f"Agent(retriever={self.retriever}, llm={self.llm})"
