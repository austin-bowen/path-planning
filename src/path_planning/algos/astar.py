from collections import defaultdict
from collections.abc import Callable
from math import inf

import numpy as np

from path_planning.envs.env import Env, Node
from path_planning.envs.grid_world import GridWorld
from path_planning.utils import PriorityQueue


class AStar:
    def __init__(
        self,
        heuristic: Callable[[Node, Node], float] = None,
    ):
        self.heuristic = heuristic or euclidean_heuristic

    def search(self, env: Env, start: Node, goal: Node) -> list[Node]:
        # Nodes to explore
        open_queue: PriorityQueue[np.ndarray] = PriorityQueue()
        open_queue.push(start, 0)
        open_set: set[Node] = {start}

        # Current lowest cost from start to node
        g_score = defaultdict(lambda: inf)
        g_score[start] = 0.0

        # Estimated total cost from start to goal through node.
        # f_score[n] = g_score[n] + h(n)
        f_score = {start: self.heuristic(start, goal)}

        parents: dict[Node, Node | None] = {start: None}

        for node in open_queue:
            open_set.remove(node)

            if node == goal:
                return self._reconstruct_path(parents, node)

            for neighbor in env.get_neighbors(node):
                tentative_g_score = g_score[node] + env.get_cost(node, neighbor)
                if tentative_g_score >= g_score[neighbor]:
                    continue

                parents[neighbor] = node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                open_queue.push(neighbor, f_score[neighbor])
                open_set.add(neighbor)

        raise NoPathError(f"No path exists from {start} to {goal}")

    def _reconstruct_path(self, parents: dict[Node, Node], goal: Node) -> list[Node]:
        path = [goal]
        while (parent := parents[goal]) is not None:
            path.append(parent)
            goal = parent
        path.reverse()
        return path


def euclidean_heuristic(point0: np.ndarray, point1: np.ndarray) -> float:
    point0 = np.array(point0)
    point1 = np.array(point1)
    return np.linalg.norm(point0 - point1).item()


class NoPathError(Exception):
    pass


def main() -> None:
    env = GridWorld(
        np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    )
    a_star = AStar()
    path = a_star.search(env, (0, 0), (4, 4))

    print(path)


if __name__ == "__main__":
    main()
