from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional


class EpisodicStore:
    """
    Append-only episodic memory with simple window retrieval and TTL per category.
    Backwards-compatible helpers: add() and topk() used by tests.
    """

    def __init__(self, max_len: int = 2000, ttl_by_type: Optional[Dict[str, int]] = None):
        self.max_len = max_len
        self.events = deque(maxlen=self.max_len)
        # ttl_by_type: days per event type/category; default 30 if not specified
        self.ttl_by_type = ttl_by_type or {}

    def __repr__(self):
        return f"EpisodicStore(max_len={self.max_len}, num_events={self.__len__()})"

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, idx):
        return self.events[idx]

    def __contains__(self, event):
        return event in self.events

    # --- New API per plan ---
    def log(self, event: Dict) -> None:
        """Validate minimal keys and append. Adds expires_at based on type if missing."""
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        # minimal validation: task_id optional for tests; text recommended
        etype = event.get("type") or event.get("cat") or "note"
        # set timestamp if missing
        if "ts" not in event:
            event["ts"] = datetime.utcnow().isoformat() + "Z"
        # set expires_at based on ttl_by_type
        ttl_days = self.ttl_by_type.get(etype, 30)
        if ttl_days is not None and "expires_at" not in event:
            event["expires_at"] = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat() + "Z"
        self.events.append(event)

    def fetch(
        self,
        task_id: Optional[str] = None,
        last_n: Optional[int] = None,
        since_minutes: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Retrieve events applying TTL purge, task_id and tag filters, and windowing."""
        # purge expired on read
        self._purge_expired()
        items = list(self.events)
        if task_id:
            items = [e for e in items if e.get("task_id") == task_id]
        if filters:
            def ok(e):
                # support simple tag filter and session filter like {"session": "sess_1"}
                for k, v in filters.items():
                    if k == "tags":
                        tags = e.get("tags", [])
                        # all requested tags must be in event tags
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
            items = [e for e in items if ok(e)]
        if since_minutes:
            cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
            items = [e for e in items if _parse_iso(e.get("ts")) >= cutoff]
        if last_n is not None:
            items = items[-last_n:]
        return items

    def _purge_expired(self):
        now = datetime.utcnow()
        kept = []
        for e in self.events:
            exp = e.get("expires_at")
            if exp is None:
                kept.append(e)
            else:
                try:
                    if _parse_iso(exp) > now:
                        kept.append(e)
                except Exception:
                    kept.append(e)
        self.events = deque(kept, maxlen=self.max_len)

    # --- Backwards-compatible helpers used by tests ---
    def add(self, event):
        self.log(event)

    def topk(self, k: int = 5, where: Callable[[Dict], bool] = lambda e: True):
        return [e for e in list(self.events) if where(e)][-k:]


def _parse_iso(s: Optional[str]) -> datetime:
    if not s:
        return datetime.utcnow()
    # remove Z if present
    if s.endswith("Z"):
        s = s[:-1]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.utcnow()

