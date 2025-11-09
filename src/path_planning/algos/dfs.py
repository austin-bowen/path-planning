from collections.abc import Callable

from path_planning.algos.astar import euclidean_heuristic
from path_planning.algos.utils import reconstruct_path
from path_planning.envs.env import Env, Node
from path_planning.envs.grid_world import GridWorld, GridWorldCliRenderer


class DepthFirstSearch:
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
        parents = {start: start}
        visited = {start}
        node = start
        while node != goal:
            if renderer:
                path = reconstruct_path(parents, start, node)
                renderer.render(env, start, goal, path)

            neighbors = env.get_neighbors(node)
            neighbors = [nc for nc in neighbors if nc[0] not in visited]
            if not neighbors:
                if node == start:
                    return None

                node = parents[node]
                continue

            est_costs = (c + self.heuristic(n, goal) for n, c in neighbors)
            closest_neighbor = min(zip(est_costs, neighbors))
            closest_neighbor = closest_neighbor[1][0]
            parents[closest_neighbor] = node
            node = closest_neighbor
            visited.add(node)

        return reconstruct_path(parents, start, goal)


def main() -> None:
    renderer = GridWorldCliRenderer(wait_time=0.01)
    path_planner = DepthFirstSearch()
    while True:
        env, start, goal = GridWorld.random_full_screen(obstacle_prob=0.5, directions=8)

        path = path_planner.search(env, start, goal, renderer=renderer)


if __name__ == "__main__":
    main()
