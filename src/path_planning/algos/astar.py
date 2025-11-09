from collections import defaultdict
from collections.abc import Callable
from math import inf
from time import sleep

import numpy as np

from path_planning.algos.utils import reconstruct_path
from path_planning.envs.env import Env, Node
from path_planning.envs.grid_world import GridWorld, GridWorldCliRenderer
from path_planning.utils import PrioritySet


class AStar:
    def __init__(
        self,
        heuristic: Callable[[Node, Node], float] = None,
    ):
        self.heuristic = heuristic or euclidean_heuristic

    def search(
        self,
        env: Env,
        start: Node,
        goal: Node,
        renderer=None,
    ) -> list[Node] | None:
        # Nodes to explore
        open_set: PrioritySet[np.ndarray] = PrioritySet()
        open_set.push(start, 0)

        # Current lowest cost from start to node
        g_score = defaultdict(lambda: inf)
        g_score[start] = 0.0

        # Estimated total cost from start to goal through node.
        # f_score[n] = g_score[n] + h(n)
        f_score = {start: self.heuristic(start, goal)}

        parents: dict[Node, Node] = {start: start}

        for node in open_set:
            if renderer:
                path = reconstruct_path(parents, start, node)
                renderer.render(env, start, goal, path)

            if node == goal:
                return reconstruct_path(parents, start, node)

            for neighbor, cost in env.get_neighbors(node):
                tentative_g_score = g_score[node] + cost
                if tentative_g_score >= g_score[neighbor]:
                    continue

                parents[neighbor] = node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                open_set.push(neighbor, f_score[neighbor])

        return None


def euclidean_heuristic(point0: np.ndarray, point1: np.ndarray) -> float:
    point0 = np.array(point0)
    point1 = np.array(point1)
    return np.linalg.norm(point0 - point1).item()


def main() -> None:
    renderer = GridWorldCliRenderer(wait_time=0.0)
    a_star = AStar()
    while True:
        env, start, goal = GridWorld.random_full_screen(obstacle_prob=0.3, directions=4)

        path = a_star.search(env, start, goal, renderer=renderer)
        # print(path)
        sleep(1)


if __name__ == "__main__":
    main()
