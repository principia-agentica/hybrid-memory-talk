from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from memory.hybrid_retriever import HybridRetriever
from tools.account_tools import lookup_user, reset_password
from tracing.tracer import tracer as _tracer


class Agent:
    """
    Minimal orchestration agent for the demo.
    - Logs user/tool/assistant turns into episodic memory.
    - Uses HybridRetriever to pull episodic window + semantic policy.
    - Applies a tiny rule to call tools when the user asks to reset a password
      and an email is present.
    - Generates an answer via an LLM stub (or a provided llm) and traces the run.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: Optional[Any] = None,
        *,
        task_id: Optional[str] = None,
        session: Optional[str] = None,
        tracer: Optional[Any] = None,
    ):
        self.retriever = retriever
        self.llm = llm  # optional external llm with .generate(prompt)
        self.task_id = task_id or "demo_task"
        self.session = session
        self.tracer = tracer or _tracer

    # ---------------- Public API ----------------
    def note_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Allow callers to append arbitrary events to episodic memory."""
        self.retriever.episodic.add(event)
        return {"ok": True}

    def answer(self, question: str) -> str:
        # 1) Log user turn
        self._log(self._user_turn_event(question))

        # 2) First retrieval pass
        sid = self.tracer.start_span("retrieve", inputs=question)
        ctx1 = self.retriever.retrieve(question)
        self.tracer.end_span(sid, retrieved=ctx1)

        # 3) Orchestrate tool plan
        plan = self._orchestrate(question)
        if plan.get("tool"):
            # Log tool call
            self._log(self._tool_call_event(plan["tool"], plan.get("args", {})))
            # Execute deterministically
            result: Dict[str, Any]
            if plan["tool"] == "lookup_user":
                result = lookup_user(**plan.get("args", {}))
            elif plan["tool"] == "reset_password":
                result = reset_password(**plan.get("args", {}))
            else:
                result = {"ok": False, "reason": "unknown_tool"}
            # Log tool result
            self._log(self._tool_result_event(plan["tool"], result))

        # 4) Second retrieval (includes tool results)
        sid2 = self.tracer.start_span("retrieve", inputs=question)
        ctx2 = self.retriever.retrieve(question)
        self.tracer.end_span(sid2, retrieved=ctx2)

        # 5) Answer generation (LLM stub by default)
        if self.llm and hasattr(self.llm, "generate"):
            prompt = self._build_prompt(ctx2, question)
            answer = self.llm.generate(prompt)
        else:
            answer = self._llm_stub(ctx2, question)

        # 6) Log assistant turn
        self._log(self._assistant_turn_event(answer))

        # 7) Trace final QA span
        self.tracer.record(inputs=question, retrieved=ctx2, output=answer, span="qa")
        return answer

    def __repr__(self):
        return f"Agent(retriever={self.retriever}, llm={self.llm})"

    # ---------------- Internals ----------------
    def _log(self, event: Dict[str, Any]) -> None:
        self.retriever.episodic.add(event)

    def _user_turn_event(self, text: str) -> Dict[str, Any]:
        ev = {
            "task_id": self.task_id,
            "type": "user_turn",
            "text": text,
            "payload": {"text": text},
        }
        if self.session:
            ev["session"] = self.session
        return ev

    def _assistant_turn_event(self, text: str) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": "assistant_turn",
            "text": text,
            "payload": {"text": text},
        }

    def _tool_call_event(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": "tool_call",
            "text": f"{name}({args})",
            "meta": {"tool": name, "args": args},
        }

    def _tool_result_event(self, name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        ok = bool(result.get("ok", result.get("exists", True)))
        return {
            "task_id": self.task_id,
            "type": "tool_result",
            "text": f"{name} -> {result}",
            "meta": {"tool": name, "ok": ok, "result": result},
        }

    def _orchestrate(self, question: str) -> Dict[str, Any]:
        """Very small rule engine:
        - If the user asks to reset a password and we find an email, call lookup_user.
        - If verified, call reset_password.
        Returns a plan dict like {tool: str|None, args: {}}.
        """
        q = question.lower()
        wants_reset = any(tok in q for tok in ["reset", "recuperar", "olvidé", "forgot", "password", "contraseña"]) and (
            "reset" in q or "recuper" in q or "forgot" in q or "olvid" in q
        )
        email = self._extract_email(question)
        if not (wants_reset and email):
            return {"tool": None}
        # First check user
        info = lookup_user(email=email)
        if not info.get("exists"):
            return {"tool": "lookup_user", "args": {"email": email}}
        if not info.get("verified"):
            return {"tool": "lookup_user", "args": {"email": email}}
        # Verified user → reset
        return {"tool": "reset_password", "args": {"email": email}}

    @staticmethod
    def _extract_email(text: str) -> Optional[str]:
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return m.group(0) if m else None

    @staticmethod
    def _build_prompt(ctx_items: List[Dict[str, Any]], question: str) -> str:
        lines = ["Contexto:"]
        for it in ctx_items:
            lines.append(f"- {it.get('text','')}")
        lines.append("")
        lines.append(f"Pregunta: {question}")
        return "\n".join(lines)

    @staticmethod
    def _llm_stub(ctx_items: List[Dict[str, Any]], question: str) -> str:
        epis = [c for c in ctx_items if c.get("kind") == "episodic"][-2:]
        sems = [c for c in ctx_items if c.get("kind") == "semantic"][:2]
        lines = [
            f"User asked: {question}",
            "Key recent events:",
            *[f"- {e.get('text','')}" for e in epis],
            "Relevant policy:",
            *[f"- {s.get('text','')} (source: {s.get('source','doc#section')})" for s in sems],
            "",
            "Response:",
            "It looks like you want to reset your password. If your email is verified, you'll receive a reset link.",
            "",
            "Internal checklist:",
            "- Confirm email on file",
            "- Follow policy step 2 if unverified",
        ]
        return "\n".join(lines)
