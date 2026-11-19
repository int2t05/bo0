"""以 scikit-optimize 的 Branin 示例比较 BO 与等预算随机搜索。"""

from __future__ import annotations

from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import skopt
from skopt import dummy_minimize, gp_minimize
from skopt.benchmarks import branin


BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]
SEEDS = range(5)
CALLS = 30
KNOWN_MINIMUM = 0.397887
OUTPUT = Path(__file__).parents[1] / "assets" / "practice" / "bo-branin.png"


def best_so_far(values: list[float]) -> np.ndarray:
    """返回每次评价后观测到的最好值。"""
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def main() -> None:
    """在固定目标评价预算下比较 GP 贝叶斯优化与随机搜索。"""
    gp_histories: list[np.ndarray] = []
    random_histories: list[np.ndarray] = []

    for seed in SEEDS:
        gp_result = gp_minimize(
            branin, BOUNDS, n_calls=CALLS, n_initial_points=8, random_state=seed
        )
        random_result = dummy_minimize(
            branin, BOUNDS, n_calls=CALLS, random_state=seed
        )
        gp_histories.append(best_so_far(gp_result.func_vals))
        random_histories.append(best_so_far(random_result.func_vals))

    gp_matrix = np.vstack(gp_histories)
    random_matrix = np.vstack(random_histories)
    gp_median = np.median(gp_matrix, axis=0)
    random_median = np.median(random_matrix, axis=0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    evaluations = np.arange(1, CALLS + 1)
    ax.plot(evaluations, gp_median, color="#087e8b", linewidth=2, label="GP Bayesian optimization")
    ax.plot(evaluations, random_median, color="#d97706", linewidth=2, label="Random search")
    ax.axhline(KNOWN_MINIMUM, color="#555555", linestyle="--", label="Known minimum ≈ 0.397887")
    ax.set(xlabel="Objective evaluations", ylabel="Best observed Branin value")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)

    print(f"scikit-optimize={skopt.__version__}")
    print(f"seeds={list(SEEDS)}, evaluations_per_run={CALLS}")
    print(f"known_minimum={KNOWN_MINIMUM}")
    print(f"GP final_by_seed={[round(float(row[-1]), 6) for row in gp_matrix]}")
    print(f"random_final_by_seed={[round(float(row[-1]), 6) for row in random_matrix]}")
    print(f"GP median_final={median([float(row[-1]) for row in gp_matrix]):.6f}")
    print(f"random median_final={median([float(row[-1]) for row in random_matrix]):.6f}")
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
