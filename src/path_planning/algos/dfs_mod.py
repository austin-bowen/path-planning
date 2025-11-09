from collections.abc import Callable
from time import sleep

from path_planning.algos.astar import euclidean_heuristic
from path_planning.algos.utils import reconstruct_path
from path_planning.envs.env import Env, Node
from path_planning.envs.grid_world import GridWorld, GridWorldCliRenderer


class DepthFirstSearchMod:
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
        margin: float = 10,
        margin2: float = 10,
    ) -> list[Node] | None:
        parents = {start: start}
        visited = {start}
        node = start
        closest = self.heuristic(start, goal)
        while node != goal:
            # if renderer:
            #     path = reconstruct_path(parents, start, node)
            #     renderer.render(env, start, goal, path)

            neighbors = env.get_neighbors(node)

            neighbors = (
                # (n, c + self.heuristic(n, goal))
                (n, c, self.heuristic(n, goal))
                for n, c in neighbors
                if n not in visited
            )

            node_cost = self.heuristic(node, goal)
            if node_cost <= closest + margin or node_cost <= margin2:
                neighbors = [(c + h, h, n) for n, c, h in neighbors]
            else:
                neighbors = [(c + h, h, n) for n, c, h in neighbors if h <= node_cost]

            if not neighbors:
                if node == start:
                    return None

                node = parents[node]
                continue

            if renderer:
                path = reconstruct_path(parents, start, node)
                renderer.render(env, start, goal, path)

            closest_neighbor = min(neighbors)
            _, h, closest_neighbor = closest_neighbor
            closest = min(closest, h)
            parents[closest_neighbor] = node
            node = closest_neighbor
            visited.add(node)

        return reconstruct_path(parents, start, goal)


def main() -> None:
    renderer = GridWorldCliRenderer(wait_time=0.05)
    path_planner = DepthFirstSearchMod()
    while True:
        env, start, goal = GridWorld.random_full_screen(obstacle_prob=0.4, directions=4)

        path = path_planner.search(env, start, goal, renderer=renderer)
        sleep(1)


if __name__ == "__main__":
    main()
