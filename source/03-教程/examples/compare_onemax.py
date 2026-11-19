"""在固定评价预算下比较 DEAP GA 与均匀随机搜索。"""

from __future__ import annotations

import random
from statistics import mean, median

from ga_onemax_deap import BIT_COUNT, run_ga


RUN_SEEDS = range(5)
BUDGET = 5000


def run_random_search(seed: int, budget: int) -> dict[str, int]:
    """按预算均匀抽取位串，返回最好 OneMax 分数。"""
    rng = random.Random(seed)
    best = 0
    for _ in range(budget):
        score = sum(rng.randint(0, 1) for _ in range(BIT_COUNT))
        best = max(best, score)
    return {"seed": seed, "best_fitness": best, "evaluations": budget}


def summarize(name: str, scores: list[int]) -> None:
    """输出简单统计与到达理论最优值的比例。"""
    successes = sum(score == BIT_COUNT for score in scores)
    print(
        f"{name}: mean={mean(scores):.2f}, median={median(scores):.2f}, "
        f"success={successes}/{len(scores)}, scores={scores}"
    )


def main() -> None:
    """运行同预算的两种搜索并保留每个随机种子的结果。"""
    ga_scores = [run_ga(seed=seed, budget=BUDGET)["best_fitness"] for seed in RUN_SEEDS]
    random_scores = [
        run_random_search(seed=seed + 1000, budget=BUDGET)["best_fitness"]
        for seed in RUN_SEEDS
    ]
    print(f"bits={BIT_COUNT}, evaluations_per_run={BUDGET}, runs={len(RUN_SEEDS)}")
    print(f"known_optimum={BIT_COUNT}; random_initial_mean={BIT_COUNT / 2:.1f}")
    summarize("GA", [int(score) for score in ga_scores])
    summarize("random_search", random_scores)


if __name__ == "__main__":
    main()
