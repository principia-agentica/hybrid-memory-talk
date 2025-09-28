"""
Minimal Tracer: records inputs, retrieved ids, output, and latency to JSONL.
Aligned with context/plan.md to keep the demo simple and teachable.

Usage patterns:

from tracing.tracer import tracer, Tracer

# One-shot convenience
tracer.record(inputs=user_question, retrieved=context_items, output=answer, span="qa")

# Manual span
sid = tracer.start_span("retrieve", inputs=user_question)
# ... do work, get retrieved items and output if any ...
tracer.end_span(sid, retrieved=context_items, output=None)

Implementation notes:
- Writes JSONL rows with: {ts, span, input_len, ctx_len, retrieved_ids, output_len, latency_ms}
- "retrieved_ids" heuristic: for each item use item["id"] or item["source"] or item["provenance"],
  else fall back to str(item)[:80]. If a plain string is provided, use it as-is.
- Path is configurable; default is out/traces.jsonl. The directory will be created if missing.
- Safe: exceptions while tracing are swallowed to avoid impacting the demo.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union


JSONable = Union[dict, list, str, int, float, bool, None]


@dataclass
class _Span:
    name: str
    t0_ns: int
    inputs: Optional[str] = None
    ctx: Optional[Iterable[Any]] = None


class Tracer:
    def __init__(self, path: str = "out/traces.jsonl", enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._spans: Dict[str, _Span] = {}
        # Prepare directory lazily when first write happens

    # --------------- Public API ---------------
    def start_span(self, name: str, inputs: Optional[str] = None, ctx: Optional[Iterable[Any]] = None) -> str:
        """Start a span and return its id (a simple incrementing token).
        We use a monotonic counter based on time to avoid extra deps.
        """
        if not self.enabled:
            return "disabled"
        sid = str(time.perf_counter_ns())
        self._spans[sid] = _Span(name=name, t0_ns=time.perf_counter_ns(), inputs=inputs, ctx=ctx)
        return sid

    def end_span(
        self,
        span_id: str,
        *,
        output: Optional[str] = None,
        retrieved: Optional[Iterable[Any]] = None,
    ) -> None:
        if not self.enabled or span_id == "disabled":
            return
        span = self._spans.pop(span_id, None)
        if not span:
            return  # unknown span; ignore
        try:
            t1_ns = time.perf_counter_ns()
            latency_ms = max(0, (t1_ns - span.t0_ns) / 1_000_000.0)
            inputs = span.inputs
            ctx_iter = retrieved if retrieved is not None else span.ctx
            retrieved_ids = self._normalize_retrieved_ids(ctx_iter)
            row = {
                "ts": self._timestamp_iso(),
                "span": span.name,
                "input_len": len(inputs) if isinstance(inputs, str) else 0,
                "ctx_len": len(retrieved_ids) if retrieved_ids is not None else 0,
                "retrieved_ids": retrieved_ids if retrieved_ids is not None else [],
                "output_len": len(output) if isinstance(output, str) else 0,
                "latency_ms": round(latency_ms, 3),
            }
            self._append_row(row)
        except Exception:
            # Swallow tracing errors to not affect the demo
            return

    def record(
        self,
        *,
        inputs: Optional[str],
        retrieved: Optional[Iterable[Any]],
        output: Optional[str],
        span: str = "run",
    ) -> None:
        """Convenience one-shot record with its own span for latency measurement."""
        if not self.enabled:
            return
        sid = self.start_span(span, inputs=inputs, ctx=retrieved)
        # minimal simulated work boundary: we immediately end; if the caller wants precise latency,
        # they should use start_span/end_span around the actual workload.
        self.end_span(sid, output=output, retrieved=retrieved)

    # --------------- Internals ---------------
    def _append_row(self, row: Dict[str, JSONable]) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            line = json.dumps(row, ensure_ascii=False)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Don't raise from tracing
            pass

    def _normalize_retrieved_ids(self, items: Optional[Iterable[Any]]) -> Optional[List[str]]:
        if items is None:
            return []
        out: List[str] = []
        try:
            for it in items:
                # Common shapes: dict with id/source/provenance; or simple string
                if isinstance(it, str):
                    out.append(it)
                    continue
                if isinstance(it, dict):
                    rid = (
                        str(it.get("id"))
                        if it.get("id") is not None
                        else (
                            str(it.get("source"))
                            if it.get("source") is not None
                            else (
                                str(it.get("provenance")) if it.get("provenance") is not None else None
                            )
                        )
                    )
                    if rid is None:
                        # try nested metadata
                        md = it.get("metadata") or {}
                        rid = str(md.get("id") or md.get("source") or md.get("section") or "?")
                    out.append(rid)
                else:
                    out.append(str(it))
        except Exception:
            return None
        return out

    def _timestamp_iso(self) -> str:
        # Use local time ISO-like without timezone dependency
        t = time.time()
        lt = time.localtime(t)
        ms = int((t - int(t)) * 1000)
        return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f".{ms:03d}"


# Default tracer instance for convenience imports
tracer = Tracer()

__all__ = ["Tracer", "tracer"]
