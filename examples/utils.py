from __future__ import annotations

import time
from typing import Protocol


class _Retriever(Protocol):
    def retrieve(self, query: str):
        ...


def p95_latency_ms(retriever: _Retriever, query: str, runs: int = 20) -> float:
    """Measure retrieval latency over multiple runs and return P95 in milliseconds.

    - Uses time.perf_counter_ns() for best-available monotonic clock.
    - Runs the retriever.retrieve(query) `runs` times (default 20) and returns the
      95th percentile latency.
    - Keeps implementation tiny and dependency-free for the educational demo.
    """
    if runs <= 0:
        return 0.0
    vals = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        _ = retriever.retrieve(query)
        t1 = time.perf_counter_ns()
        vals.append((t1 - t0) / 1_000_000.0)
    vals.sort()
    idx = max(0, int(round(0.95 * (len(vals) - 1))))
    return vals[idx]


__all__ = ["p95_latency_ms"]
