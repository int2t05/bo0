"""改编 DEAP 官方 OneMax 示例，运行一个固定预算的遗传算法。"""

from __future__ import annotations

import random
from typing import Any

from deap import base, creator, tools


BIT_COUNT = 100
POPULATION_SIZE = 100

if not hasattr(creator, "FitnessMaxTutorial"):
    creator.create("FitnessMaxTutorial", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualTutorial"):
    creator.create("IndividualTutorial", list, fitness=creator.FitnessMaxTutorial)


def build_toolbox() -> base.Toolbox:
    """建立 OneMax 的个体、评价与遗传算子。"""
    toolbox = base.Toolbox()
    toolbox.register("bit", random.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.IndividualTutorial, toolbox.bit, n=BIT_COUNT
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda individual: (sum(individual),))
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / BIT_COUNT)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def run_ga(seed: int = 7, budget: int = 5000) -> dict[str, Any]:
    """在固定目标评价预算下运行 GA，并返回最优个体与评价次数。"""
    if budget < POPULATION_SIZE:
        raise ValueError(f"预算至少应为初始种群大小 {POPULATION_SIZE}")

    random.seed(seed)
    toolbox = build_toolbox()
    population = toolbox.population(n=POPULATION_SIZE)
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    evaluations = len(population)
    generations = 0

    while evaluations < budget:
        generations += 1
        parents = toolbox.select(population, k=POPULATION_SIZE)
        offspring = [toolbox.clone(parent) for parent in parents]

        for child_a, child_b in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child_a, child_b)
                del child_a.fitness.values
                del child_b.fitness.values

        for child in offspring:
            if random.random() < 0.2:
                toolbox.mutate(child)
                del child.fitness.values

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        if not invalid:
            # 全部子代都未变化时，强制变异一个个体，避免评价预算停滞。
            child = toolbox.clone(random.choice(population))
            toolbox.mutate(child)
            del child.fitness.values
            invalid = [child]

        invalid = invalid[: budget - evaluations]
        for individual in invalid:
            individual.fitness.values = toolbox.evaluate(individual)
        evaluations += len(invalid)
        population = tools.selBest(population + invalid, k=POPULATION_SIZE)

    best = tools.selBest(population, k=1)[0]
    return {
        "seed": seed,
        "best_fitness": int(best.fitness.values[0]),
        "best_individual": "".join(str(bit) for bit in best),
        "evaluations": evaluations,
        "generations": generations,
    }


def main() -> None:
    """运行单种子演示，并打印可核对摘要。"""
    result = run_ga()
    print(f"seed={result['seed']}")
    print(f"evaluations={result['evaluations']}")
    print(f"generations={result['generations']}")
    print(f"best_fitness={result['best_fitness']} / {BIT_COUNT}")
    all_ones = set(result["best_individual"]) == {"1"}
    print(f"all_ones={all_ones}")


if __name__ == "__main__":
    main()
