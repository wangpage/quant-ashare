from .parser import Level2Parser, TradeTick, OrderTick, OrderBookSnapshot
from .buffer import RingBuffer
from .simulator import Level2Simulator

# nats_client 只在 nats-py 装了才能用, 懒加载
def __getattr__(name):
    if name == "Level2NatsClient":
        from .nats_client import Level2NatsClient
        return Level2NatsClient
    raise AttributeError(name)

__all__ = [
    "Level2NatsClient",
    "Level2Parser",
    "Level2Simulator",
    "TradeTick",
    "OrderTick",
    "OrderBookSnapshot",
    "RingBuffer",
]
