"""按公开 ACO Lab 的图结构实现一个最短路蚁群示例。"""

from __future__ import annotations

import random
from collections import defaultdict


EDGES = {
    (1, 2): 4,
    (1, 3): 3,
    (2, 3): 5,
    (2, 4): 4,
    (3, 4): 2,
    (2, 5): 1,
    (4, 5): 6,
}
START = 1
GOAL = 5


def make_graph() -> tuple[dict[int, list[int]], dict[tuple[int, int], int]]:
    """把无向边表转换成邻接表和可双向查询的长度表。"""
    neighbors: dict[int, list[int]] = defaultdict(list)
    lengths: dict[tuple[int, int], int] = {}
    for (left, right), distance in EDGES.items():
        neighbors[left].append(right)
        neighbors[right].append(left)
        lengths[left, right] = distance
        lengths[right, left] = distance
    return dict(neighbors), lengths


def path_length(path: list[int], lengths: dict[tuple[int, int], int]) -> int:
    """计算路径长度；遇到不存在的边时抛出错误。"""
    return sum(lengths[left, right] for left, right in zip(path, path[1:]))


def exact_shortest_path(
    neighbors: dict[int, list[int]], lengths: dict[tuple[int, int], int]
) -> tuple[list[int], int]:
    """枚举所有简单路径，给小图计算精确基线。"""
    paths: list[list[int]] = []

    def visit(node: int, path: list[int]) -> None:
        if node == GOAL:
            paths.append(path[:])
            return
        for next_node in neighbors[node]:
            if next_node not in path:
                visit(next_node, path + [next_node])

    visit(START, [START])
    best = min(paths, key=lambda path: path_length(path, lengths))
    return best, path_length(best, lengths)


def choose_next(
    current: int,
    visited: list[int],
    neighbors: dict[int, list[int]],
    lengths: dict[tuple[int, int], int],
    pheromone: dict[tuple[int, int], float],
    rng: random.Random,
) -> int | None:
    """按信息素与距离倒数做轮盘赌选择。"""
    candidates = [node for node in neighbors[current] if node not in visited]
    if not candidates:
        return None
    weights = [pheromone[current, node] / lengths[current, node] for node in candidates]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for node, weight in zip(candidates, weights):
        cumulative += weight
        if cumulative >= threshold:
            return node
    return candidates[-1]


def run_aco(seed: int = 10, ants: int = 20, iterations: int = 100) -> dict[str, object]:
    """运行简化 Ant System，返回找到的最好路径。"""
    rng = random.Random(seed)
    neighbors, lengths = make_graph()
    pheromone = {edge: 0.5 for edge in lengths}
    evaporation = 0.7
    best_path: list[int] | None = None
    best_cost: int | None = None

    for _ in range(iterations):
        paths: list[list[int]] = []
        for _ in range(ants):
            path = [START]
            while path[-1] != GOAL:
                next_node = choose_next(
                    path[-1], path, neighbors, lengths, pheromone, rng
                )
                if next_node is None:
                    break
                path.append(next_node)
            if path[-1] == GOAL:
                paths.append(path)

        for edge in pheromone:
            pheromone[edge] *= 1.0 - evaporation
        for path in paths:
            cost = path_length(path, lengths)
            if best_cost is None or cost < best_cost:
                best_path, best_cost = path, cost
            deposit = 1.0 / cost
            for left, right in zip(path, path[1:]):
                pheromone[left, right] += deposit
                pheromone[right, left] += deposit

    if best_path is None or best_cost is None:
        raise RuntimeError("本次运行没有构造出从起点到终点的路径")
    return {"seed": seed, "path": best_path, "cost": best_cost}


def main() -> None:
    """运行蚁群搜索并与精确枚举对照。"""
    neighbors, lengths = make_graph()
    exact_path, exact_cost = exact_shortest_path(neighbors, lengths)
    result = run_aco()
    path = result["path"]
    assert isinstance(path, list)
    assert path[0] == START and path[-1] == GOAL
    assert len(path) == len(set(path))
    assert all(edge in lengths for edge in zip(path, path[1:]))
    print(f"seed={result['seed']}")
    print(f"aco_path={result['path']}, cost={result['cost']}")
    print(f"exact_path={exact_path}, cost={exact_cost}")


if __name__ == "__main__":
    main()
