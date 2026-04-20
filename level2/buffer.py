"""高速环形缓冲 - 用于 Level2 消息落盘 + 策略消费."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable


class RingBuffer:
    """线程安全的环形缓冲.

    生产者 (NATS回调) 往里写, 消费者 (策略/存盘) 批量取.
    超过容量时按 FIFO 丢弃最旧的.
    """

    def __init__(self, maxlen: int = 100_000):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._pushed = 0
        self._dropped = 0

    def push(self, item: Any) -> None:
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(item)
            self._pushed += 1

    def pop_batch(self, n: int = 1000) -> list[Any]:
        out = []
        with self._lock:
            for _ in range(min(n, len(self._buf))):
                out.append(self._buf.popleft())
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._buf),
                "capacity": self._buf.maxlen,
                "pushed": self._pushed,
                "dropped": self._dropped,
                "drop_rate": self._dropped / max(self._pushed, 1),
            }


class BatchConsumer(threading.Thread):
    """独立线程, 批量消费 RingBuffer 并调用 sink."""

    def __init__(
        self,
        buf: RingBuffer,
        sink: Callable[[list[Any]], None],
        batch_size: int = 1000,
        poll_sleep_ms: int = 5,
        name: str = "BatchConsumer",
    ):
        super().__init__(name=name, daemon=True)
        self.buf = buf
        self.sink = sink
        self.batch_size = batch_size
        self.poll_sleep_ms = poll_sleep_ms
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        import time
        while not self._stop_flag.is_set():
            items = self.buf.pop_batch(self.batch_size)
            if items:
                try:
                    self.sink(items)
                except Exception as e:
                    print(f"[{self.name}] sink error: {e}")
            else:
                time.sleep(self.poll_sleep_ms / 1000)
