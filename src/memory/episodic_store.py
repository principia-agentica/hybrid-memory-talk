from collections import deque


class EpisodicStore:
    def __init__(self, max_len=2000):
        self.max_len = max_len
        self.events = deque(maxlen=self.max_len)

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

    def add(self, event):
        self.events.append(event)

    def topk(self, k=5, where=lambda e: True):
        return [e for e in list(self.events)[-k:] if where(e)]

