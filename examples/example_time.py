import time
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from kinematics.config.doosan_m0609 import DoosanM0609Config
from kinematics.core import RobotArmKinematics


def benchmark_numba(
    func: Callable[..., Any],
    *args: Any,
    warmup: int = 1,
    repeat: int = 100,
    number: int = 1000,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Benchmark a Numba function.

    The first call compiles the function (JIT), then `warmup` executions are
    performed before timing `repeat` executions.

    Returns
    -------
    dict
        {
            "mean_ms": ...,
            "std_ms": ...,
            "min_ms": ...,
            "max_ms": ...
        }
    """

    # First call -> compilation
    func(*args, **kwargs)

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []

    for _ in range(repeat):
        start = time.perf_counter()

        for _ in range(number):
            func(*args, **kwargs)

        elapsed = (time.perf_counter() - start) / number
        times.append(elapsed * 1000)  # ms per execution

    return {
        "mean_ms": mean(times),
        "std_ms": stdev(times) if repeat > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main() -> None:
    config = DoosanM0609Config()
    robot = RobotArmKinematics(config)

    # # tcp pose
    # tcp = np.identity(4)
    # tcp[2, 3] = 0.2759
    # config.tcp = tcp
    # # tool shape
    # config.set_tool_shape(1.280, np.array([-0.01396, 0.0067, 0.195]))

    q = np.random.uniform(config.q_min, config.q_max)  # random

    # test forward kinematic
    stats = benchmark_numba(
        robot.fk,
        q,
        warmup=1,
        repeat=50,
        number=1000,
    )

    print("Stats forward kinematics :")
    print(f"Mean : {stats['mean_ms']:.4f} ms")
    print(f"Std  : {stats['std_ms']:.4f} ms")
    print(f"Min  : {stats['min_ms']:.4f} ms")
    print(f"Max  : {stats['max_ms']:.4f} ms")

    # test jacobian
    stats = benchmark_numba(
        robot.get_jacobian,
        q,
        warmup=1,
        repeat=50,
        number=1000,
    )

    print("Stats jacobian :")
    print(f"Mean : {stats['mean_ms']:.4f} ms")
    print(f"Std  : {stats['std_ms']:.4f} ms")
    print(f"Min  : {stats['min_ms']:.4f} ms")
    print(f"Max  : {stats['max_ms']:.4f} ms")

    # test inverse kinematics
    # position = np.array([0.6, 0.6, 0.4])
    position = np.array([0.2, 0.4, 0.2])
    rot = R.from_euler("ZYZ", [0, 90, 90], degrees=True)
    pose = np.identity(4)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = position.T

    q_init = np.array(
        [-3.08334423, 0.54545424, -1.92686675, 2.03527711, -1.36723453, 4.2063956]
    )

    stats = benchmark_numba(
        robot.ik,
        pose,
        q_init,
        warmup=1,
        repeat=5,
        number=100,
    )

    print("Stats inverse kinematics :")
    print(f"Mean : {stats['mean_ms']:.4f} ms")
    print(f"Std  : {stats['std_ms']:.4f} ms")
    print(f"Min  : {stats['min_ms']:.4f} ms")
    print(f"Max  : {stats['max_ms']:.4f} ms")


if __name__ == "__main__":
    main()
