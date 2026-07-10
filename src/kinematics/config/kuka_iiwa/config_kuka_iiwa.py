import numpy as np

from kinematics.config.config import BaseConfig, RepresentationType


class KukaIiwaConfig(BaseConfig):
    def __init__(self) -> None:
        a = np.array([0, 0, 0, 0, 0, 0, 0])
        d = np.array([0.360, 0, 0.420, 0, 0.400, 0, 0.126])
        alpha = np.array(
            [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2]
        )
        theta = np.array([0, 0, 0, 0, 0, 0, 0])

        representation = RepresentationType.DH_KHALIL

        q_min = np.deg2rad(
            np.array([-170, -120, -170, -120, -170, -120, -175], dtype=np.float64)
        )
        q_max = np.deg2rad(
            np.array([170, 120, 170, 120, 170, 120, 175], dtype=np.float64)
        )

        super().__init__(a, d, alpha, theta, representation, q_min, q_max)
