from path_planning.envs.env import Node


def reconstruct_path(parents: dict[Node, Node], start: Node, goal: Node) -> list[Node]:
    path = [goal]
    while (parent := parents[goal]) != start:
        path.append(parent)
        goal = parent
    path.reverse()
    return path
