"""
Centralized configuration with sensible defaults and optional environment overrides.

This module exposes simple constants for the demo defaults described in context/plan.md.
Users can override them via environment variables without changing code.

Environment variables (all optional):
- HM_K_EPI: int, number of episodic events to retrieve (default 4)
- HM_K_SEM: int, number of semantic items to retrieve (default 3)
- HM_TOKEN_BUDGET: int, token budget for the merged context (default 1600)
- HM_EPISODIC_TTL_DAYS: int, default TTL in days for episodic events when not
  specified per-type (default 30)
- HM_EPI_FILTERS_JSON: JSON dict with simple equality/tag filters to apply when
  fetching episodic window (default {"session": None} → effectively no-op)
- HM_SEM_FILTERS_JSON: JSON dict with metadata filters for semantic search
  (default {"tags": ["policy"], "pii": False})
- HM_RERANKER_ENABLED: bool flag ("1", "true", "yes" enable) default False

Notes:
- Filters are intentionally simple: equality on top-level keys and special-case
  "tags" as a list-contains-all semantics. The HybridRetriever will build a
  predicate from HM_EPI_FILTERS_JSON when the user does not pass an epi_filter.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    return val in ("1", "true", "yes", "on")


def _get_json(name: str, default: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    val = os.getenv(name)
    if not val:
        return default
    try:
        obj = json.loads(val)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return default


# Defaults aligned with the plan
K_EPI: int = _get_int("HM_K_EPI", 4)
K_SEM: int = _get_int("HM_K_SEM", 3)
TOKEN_BUDGET: int = _get_int("HM_TOKEN_BUDGET", 1600)
EPISODIC_TTL_DAYS: int = _get_int("HM_EPISODIC_TTL_DAYS", 30)

# Filters
EPI_FILTERS: Optional[Dict[str, Any]] = _get_json("HM_EPI_FILTERS_JSON", None)
SEM_FILTERS: Optional[Dict[str, Any]] = _get_json(
    "HM_SEM_FILTERS_JSON", {"tags": ["policy"], "pii": False}
)

# Flags
RERANKER_ENABLED: bool = _get_bool("HM_RERANKER_ENABLED", False)


__all__ = [
    "K_EPI",
    "K_SEM",
    "TOKEN_BUDGET",
    "EPISODIC_TTL_DAYS",
    "EPI_FILTERS",
    "SEM_FILTERS",
    "RERANKER_ENABLED",
]
