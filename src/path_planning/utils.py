import heapq
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    @dataclass(order=True, slots=True)
    class _Node:
        item: Any = field(compare=False)
        priority: float

    def __init__(self):
        self.queue: list[PriorityQueue._Node] = []

    def __iter__(self):
        while not self.is_empty():
            yield self.pop()

    def push(self, item: T, priority: float) -> None:
        heapq.heappush(self.queue, PriorityQueue._Node(item, priority))

    def pop(self) -> T:
        try:
            return heapq.heappop(self.queue).item
        except IndexError:
            raise IndexError("pop from empty queue")

    def is_empty(self):
        return len(self.queue) == 0
