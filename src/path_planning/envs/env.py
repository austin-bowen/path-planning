from abc import ABC
from collections.abc import Iterable
from typing import Any

Node = Any
Cost = float


class Env(ABC):
    def get_neighbors(self, node: Node) -> Iterable[tuple[Node, Cost]]: ...
