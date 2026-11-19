"""改编 PySwarms 官方函数优化示例，最小化 Sphere 函数。"""

from __future__ import annotations

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj


def main() -> None:
    """用固定随机种子运行二维全局最优 PSO。"""
    seed = 7
    np.random.seed(seed)
    lower = np.array([-5.0, -5.0])
    upper = np.array([5.0, 5.0])
    initial_positions = np.random.uniform(lower, upper, size=(20, 2))
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=20,
        dimensions=2,
        options=options,
        bounds=(lower, upper),
        init_pos=initial_positions,
    )
    cost, position = optimizer.optimize(
        single_obj.sphere, iters=250, verbose=False
    )

    print(f"seed={seed}")
    print(f"iterations=250, particles=20")
    print(f"best_cost={cost:.12g}")
    print(f"best_position={position.tolist()}")
    print(f"direct_check={float(np.sum(position ** 2)):.12g}")


if __name__ == "__main__":
    main()
