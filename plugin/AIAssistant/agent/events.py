"""Event log shared by the agent, the tools and the chat page.

Everything the user should see — messages, tool calls and their results,
pipeline stages, slicing progress — is appended here with an increasing
sequence number. The page long-polls `/api/events?since=N`.
"""
import threading
import time
from collections import OrderedDict


class EventLog(object):
    def __init__(self, limit=5000):
        self._events = []
        self._seq = 0
        self._limit = limit
        self._cond = threading.Condition()

    def emit(self, kind, **data):
        with self._cond:
            self._seq += 1
            event = OrderedDict([("seq", self._seq), ("t", round(time.time(), 3)), ("kind", kind)])
            event.update(data)
            self._events.append(event)
            if len(self._events) > self._limit:
                del self._events[:len(self._events) - self._limit]
            self._cond.notify_all()
            return event

    def since(self, seq, timeout=0.0):
        """Events with a sequence number above `seq`; waits up to `timeout` s for new ones."""
        deadline = time.time() + max(0.0, timeout)
        with self._cond:
            while True:
                fresh = [e for e in self._events if e["seq"] > seq]
                remaining = deadline - time.time()
                if fresh or remaining <= 0:
                    return fresh
                self._cond.wait(remaining)

    def last_seq(self):
        with self._cond:
            return self._seq

    def clear(self):
        with self._cond:
            self._events = []
            self._cond.notify_all()
