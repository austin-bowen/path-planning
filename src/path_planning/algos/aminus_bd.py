from collections.abc import Callable
from time import sleep

from path_planning.algos.astar import euclidean_heuristic
from path_planning.algos.utils import reconstruct_path
from path_planning.envs.env import Env, Node
from path_planning.envs.grid_world import GridWorld, GridWorldCliRenderer
from path_planning.utils import PriorityQueue


class AMinus:
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
        open_set = PriorityQueue()
        open_set.push(start, 0)
        parents = {start: start}
        for node in open_set:
            if node == goal:
                yield reconstruct_path(parents, start, goal)
                return

            if renderer:
                path = reconstruct_path(parents, start, node)
                renderer.render(env, start, goal, path)

            neighbors = env.get_neighbors(node)

            for n, c in neighbors:
                if n not in parents:
                    parents[n] = node
                    h = self.heuristic(n, goal)
                    open_set.push(n, h)

            yield None

        yield "no path"


def main() -> None:
    renderer = GridWorldCliRenderer(wait_time=0.01)
    path_planner = AMinus()
    while True:
        env, start, goal = GridWorld.random_full_screen(obstacle_prob=0.4, directions=4)

        path1 = path_planner.search(env, start, goal, renderer=renderer)
        path2 = path_planner.search(env, goal, start, renderer=renderer)

        while True:
            tmp = next(path1)
            if tmp is not None:
                break

            tmp = next(path2)
            if tmp is not None:
                break

        sleep(1)


if __name__ == "__main__":
    main()
