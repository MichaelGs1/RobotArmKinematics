import numpy as np

from kinematics.config.config import BaseConfig


class UR20Config(BaseConfig):
    def __init__(self) -> None:
        d = np.array([0.236, 0, 0, 0.2019, 0.1594, 0.1548])
        a = np.array([0, 0, 0.8618, 0.7277, 0, 0])
        alpha = np.array([0, np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi, 0, 0, 0, np.pi])

        q_min = np.deg2rad(
            np.array([-360, -360, -360, -360, -360, -360], dtype=np.float64)
        )
        q_max = np.deg2rad(np.array([360, 360, 360, 360, 360, 360], dtype=np.float64))

        super().__init__(a, d, alpha, theta, q_min, q_max)
