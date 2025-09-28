import json
from pathlib import Path

from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.hybrid_retriever import HybridRetriever
from tracing.tracer import Tracer
from agent import Agent


class TinyEncoder:
    def embed(self, text: str):
        # very small deterministic vector; length-based for simplicity
        return [len(text) % 7, 1.0]


def _read_last_jsonl(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    assert text, "trace file should not be empty"
    line = text.splitlines()[-1]
    return json.loads(line)


def test_agent_flow_with_email_calls_tools_and_logs(tmp_path):
    # Stores and retriever
    epi = EpisodicStore(max_len=20)
    sem = SemanticStore(encoder=TinyEncoder())
    sem.add({
        "id": "p1",
        "text": "Password reset policy requires verified email and logs the action.",
        "metadata": {"source": "policy.md", "section": "password", "tags": ["policy"], "pii": False},
    })
    retr = HybridRetriever(episodic=epi, semantic=sem, k_epi=4, k_sem=2, reranker_enabled=False)

    # Tracer to an isolated file
    out_file = tmp_path / "traces.jsonl"
    tracer = Tracer(path=str(out_file))

    agent = Agent(retriever=retr, task_id="t1", session="s1", tracer=tracer)

    question = "I forgot my password. Please reset it for ana@example.com"
    answer = agent.answer(question)

    # Basic expectations on the answer (stubbed response)
    assert isinstance(answer, str)
    assert "Internal checklist:" in answer
    assert "Relevant policy:" in answer

    # Episodic events should include user turn, a tool call, a tool result, and assistant turn
    events = list(epi)
    types = [e.get("type") or e.get("cat") for e in events]
    assert "user_turn" in types
    assert "assistant_turn" in types
    assert "tool_call" in types, f"types logged: {types}"
    assert "tool_result" in types, f"types logged: {types}"

    # The tool_result should reference reset_password and be ok for ana@example.com (verified user)
    tool_results = [e for e in events if (e.get("type") == "tool_result")]
    assert tool_results, "expected at least one tool_result"
    tr_text = tool_results[-1].get("text", "")
    assert "reset_password" in tr_text or "lookup_user" in tr_text

    # Tracer wrote a final QA row
    assert out_file.exists()
    row = _read_last_jsonl(out_file)
    assert row["span"] in ("qa", "run")
    assert isinstance(row.get("retrieved_ids", []), list)


def test_agent_no_email_skips_tools(tmp_path):
    epi = EpisodicStore(max_len=10)
    sem = SemanticStore(encoder=TinyEncoder())
    sem.add({"id": "p2", "text": "Guidance about k and token budget.",
             "metadata": {"source": "policy.md", "section": "retrieval", "tags": ["policy"], "pii": False}})
    retr = HybridRetriever(episodic=epi, semantic=sem, k_epi=2, k_sem=1, reranker_enabled=False)

    tracer = Tracer(path=str(tmp_path / "t.jsonl"))
    agent = Agent(retriever=retr, task_id="t2", tracer=tracer)

    answer = agent.answer("General guidance on retrieval please")
    assert isinstance(answer, str)
    assert "Response:" in answer

    types = [e.get("type") or e.get("cat") for e in list(epi)]
    assert types.count("user_turn") == 1
    assert types.count("assistant_turn") == 1
    assert "tool_call" not in types
    assert "tool_result" not in types


def test_agent_note_event_appends(tmp_path):
    epi = EpisodicStore(max_len=5)
    sem = SemanticStore(encoder=TinyEncoder())
    retr = HybridRetriever(episodic=epi, semantic=sem, k_epi=1, k_sem=0)

    tracer = Tracer(path=str(tmp_path / "t2.jsonl"))
    agent = Agent(retriever=retr, task_id="t3", tracer=tracer)

    note = {"id": "n1", "type": "note", "text": "remember to follow up"}
    res = agent.note_event(note)
    assert res.get("ok") is True
    assert note in list(epi)
