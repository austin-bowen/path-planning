from itertools import product
from math import inf
from typing import Iterable

import numpy as np

from path_planning.envs.env import Env

Node = tuple[int, int]


class GridWorld(Env):
    def __init__(
        self,
        obstacle_map: np.ndarray,
        cost: float = 1.0,
    ):
        self.obstacle_map = obstacle_map
        self.cost = cost

    @property
    def width(self) -> int:
        return self.obstacle_map.shape[1]

    @property
    def height(self) -> int:
        return self.obstacle_map.shape[0]

    def get_neighbors(self, node: Node) -> Iterable[Node]:
        height, width = self.obstacle_map.shape

        for dr, dc in [
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        ]:
            if dr == 0 and dc == 0:
                continue

            row = node[0] + dr
            col = node[1] + dc

            if (
                0 <= row < height
                and 0 <= col < width
                and not self.obstacle_map[row, col]
            ):
                yield row, col

    def get_cost(self, node0: Node, node1: Node) -> float:
        distance = abs(node0[0] - node1[0]) + abs(node1[1] - node1[1])
        return self.cost if distance <= 1 else inf
