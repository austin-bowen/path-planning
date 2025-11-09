import os
import time
from random import randint
from typing import Iterable, Literal

import numpy as np

from path_planning.envs.env import Cost, Env

Node = tuple[int, int]
Directions = Literal[4, 8]


class GridWorld(Env):
    def __init__(
        self,
        obstacle_map: np.ndarray,
        directions: Directions,
    ):
        self.obstacle_map = obstacle_map
        self.directions = directions

        if directions == 4:
            self._dr_dc = [
                (-1, 0),
                (0, 1),
                (1, 0),
                (0, -1),
            ]
        elif directions == 8:
            self._dr_dc = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        else:
            raise ValueError(f"directions must be 4 or 8. directions={directions}")

        self._obstacles_seen: set[Node] = set()

    @classmethod
    def random(
        cls,
        shape: tuple[int, int],
        start: Node,
        goal: Node,
        obstacle_prob: float,
        directions: Directions,
    ) -> "GridWorld":
        obstacle_map = (np.random.random(shape) < obstacle_prob).astype(int)
        obstacle_map[start] = obstacle_map[goal] = 0
        return cls(obstacle_map, directions)

    @classmethod
    def random_full_screen(
        cls,
        obstacle_prob: float,
        directions: Directions,
    ) -> tuple["GridWorld", Node, Node]:
        term_size = os.get_terminal_size()
        r, c = term_size.lines - 4, (term_size.columns // 2) - 4

        start = (0, 0)
        goal = (r - 1, c - 1)
        # start = (randint(0, r - 1), randint(0, c - 1))
        # while (goal := (randint(0, r - 1), randint(0, c - 1))) == start:
        #     pass

        start = (r // 2, c // 5)
        goal = (r // 2, c - start[1])
        # start = (start, start)

        env = cls.random((r, c), start, goal, obstacle_prob, directions)
        return env, start, goal

    @property
    def width(self) -> int:
        return self.obstacle_map.shape[1]

    @property
    def height(self) -> int:
        return self.obstacle_map.shape[0]

    def get_neighbors(self, node: Node) -> Iterable[tuple[Node, Cost]]:
        for dr, dc in self._dr_dc:
            neighbor = (node[0] + dr, node[1] + dc)
            if self._is_open(neighbor):
                cost = self._get_cost(node, neighbor)
                yield neighbor, cost

    def _is_open(self, node: Node) -> bool:
        row, col = node

        if not (0 <= row < self.height and 0 <= col < self.width):
            return False

        is_obstacle = self.obstacle_map[row, col]
        if is_obstacle:
            self._obstacles_seen.add(node)

        return not is_obstacle

    def _get_cost(self, node0: Node, node1: Node) -> float:
        return ((node0[0] - node1[0]) ** 2 + (node0[1] - node1[1]) ** 2) ** 0.5


class GridWorldCliRenderer:
    def __init__(self, wait_time: float = 0.1):
        self.wait_time = wait_time
        self._first_render = True

    def render(
        self,
        env: GridWorld,
        start: Node = None,
        goal: Node = None,
        path: list[Node] = None,
    ) -> None:
        w, h = env.width, env.height
        cells = [[" "] * w for _ in range(h)]

        # for r in range(h):
        #     for c in range(w):
        #         if env.obstacle_map[r, c]:
        #             cells[r][c] = "█"

        for r, c in env._obstacles_seen:
            cells[r][c] = "█"

        for r, c in path or []:
            cells[r][c] = ":"

        if start:
            cells[start[0]][start[1]] = "S"
        if goal:
            cells[goal[0]][goal[1]] = "G"

        if self._first_render:
            self._first_render = False
        else:
            print(end="\r")
            print(end=f"\033[{h + 2}A")

        print("██" * (w + 2))
        for r in cells:
            print("██" + "".join(c * 2 for c in r) + "██")
        print("██" * (w + 2))

        time.sleep(self.wait_time)
