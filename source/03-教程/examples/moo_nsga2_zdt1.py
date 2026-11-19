"""运行 pymoo 官方 ZDT1 基准，并与理论 Pareto 前沿比较。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem


SEED = 7
POPULATION_SIZE = 100
GENERATIONS = 200
OUTPUT = Path(__file__).parents[1] / "assets" / "practice" / "zdt1-nsga2.png"


def main() -> None:
    """求解 ZDT1、检查前沿近似误差并保存图。"""
    problem = get_problem("zdt1")
    result = minimize(
        problem,
        NSGA2(pop_size=POPULATION_SIZE, eliminate_duplicates=True),
        ("n_gen", GENERATIONS),
        seed=SEED,
        verbose=False,
    )
    if result.F is None:
        raise RuntimeError("pymoo 未返回非支配解集")

    objectives = np.asarray(result.F)
    expected_f2 = 1.0 - np.sqrt(np.clip(objectives[:, 0], 0.0, 1.0))
    rmse = float(np.sqrt(np.mean((objectives[:, 1] - expected_f2) ** 2)))
    true_f1 = np.linspace(0.0, 1.0, 300)
    true_f2 = 1.0 - np.sqrt(true_f1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(true_f1, true_f2, color="#d97706", linewidth=2, label="Known Pareto front")
    ax.scatter(
        objectives[:, 0], objectives[:, 1],
        s=16, alpha=0.8, color="#087e8b", label="NSGA-II solutions",
    )
    ax.set(xlabel="f1", ylabel="f2", title="ZDT1: NSGA-II and known Pareto front")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)

    print(f"pymoo={pymoo.__version__}")
    print(f"seed={SEED}, population={POPULATION_SIZE}, generations={GENERATIONS}")
    print(f"evaluations={result.algorithm.evaluator.n_eval}")
    print(f"nondominated_solutions={len(objectives)}")
    print(f"front_rmse={rmse:.6f}")
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
