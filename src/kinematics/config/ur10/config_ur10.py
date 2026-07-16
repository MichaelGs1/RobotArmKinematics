import numpy as np

from kinematics.config.config import BaseConfig, RepresentationType


class UR10Config(BaseConfig):
    __slots__ = ()

    def __init__(self) -> None:
        d = np.array([0.128, 0, 0, 0.1639, 0.1157, 0.0922])
        a = np.array([0, 0, 0.6127, 0.5716, 0, 0])
        alpha = np.array([0, np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2])
        theta = np.array([0, np.pi, 0, 0, 0, np.pi])

        representation = RepresentationType.DH_KHALIL

        q_min = np.deg2rad(
            np.array([-360, -360, -360, -360, -360, -360], dtype=np.float64)
        )
        q_max = np.deg2rad(np.array([360, 360, 360, 360, 360, 360], dtype=np.float64))

        super().__init__(a, d, alpha, theta, representation, q_min, q_max)
