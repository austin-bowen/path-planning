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
        self._queue: list[PriorityQueue._Node] = []

    def __contains__(self, item: T) -> bool:
        return any(n.item == item for n in self._queue)

    def __iter__(self):
        while not self.is_empty():
            yield self.pop()

    def __len__(self) -> int:
        return len(self._queue)

    def push(self, item: T, priority: float) -> None:
        heapq.heappush(self._queue, PriorityQueue._Node(item, priority))

    def pop(self) -> T:
        try:
            return heapq.heappop(self._queue).item
        except IndexError:
            raise IndexError("pop from empty queue")

    def update(self, item: T, priority: float) -> None:
        node = next(n for n in self._queue if n.item == item)
        node.priority = priority
        heapq.heapify(self._queue)

    def is_empty(self):
        return len(self._queue) == 0


class PrioritySet(PriorityQueue, Generic[T]):
    def __init__(self):
        super().__init__()
        self._set: set[T] = set()

    def __contains__(self, item: T) -> bool:
        return item in self._set

    def push(self, item: T, priority: float) -> None:
        """
        If the item is already in the set, update its priority.
        Otherwise, add the item to the set and push it to the queue.
        """

        if item in self:
            self.update(item, priority)
        else:
            super().push(item, priority)
            self._set.add(item)

        assert len(self._set) == len(self._queue)

    def pop(self) -> T:
        item = super().pop()
        self._set.remove(item)
        assert len(self._set) == len(self._queue)
        return item
