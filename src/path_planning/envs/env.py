from abc import ABC
from collections.abc import Iterable
from typing import Any

Node = Any


class Env(ABC):
    def get_neighbors(self, node: Node) -> Iterable[Node]: ...

    def get_cost(self, node0: Node, node1: Node) -> float: ...
