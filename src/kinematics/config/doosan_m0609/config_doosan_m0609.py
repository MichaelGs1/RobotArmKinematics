import numpy as np

from kinematics.config.config import BaseConfig, RepresentationType
from kinematics.core.core import _get_link_matrix_numba


class DoosanM0609Config(BaseConfig):
    def __init__(self) -> None:
        # doosan dh
        a3 = 0.411
        d1 = 0.135
        d2 = 0.00625
        d4 = 0.368
        d6 = 0.121

        a = np.array([0, 0, a3, 0, 0, 0])
        d = np.array([d1, d2, 0, d4, 0, d6])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])

        representation = RepresentationType.DH_KHALIL

        # geom limit
        q_min = np.deg2rad(
            np.array([-360, -95, -125, -360, -135, -330], dtype=np.float64)
        )
        q_max = np.deg2rad(np.array([360, 95, 125, 360, 135, 330], dtype=np.float64))

        # dynamic limit
        q_point_max = np.deg2rad(
            np.array([150, 150, 180, 225, 225, 225], dtype=np.float64)
        )
        torque_max = np.array([160, 160, 90, 45, 45, 45], dtype=np.float64)

        # dynamic param
        masses = np.array([5.02, 8.04, 3.6, 3.57, 2.83, 1.16, 0], dtype=np.float64)
        cog = (
            np.array(
                [
                    [0.07, 36.23, 131.58],
                    [0.03, 166.3, 339.47],
                    [-0.02, 49.7, 552.87],
                    [0.04, 103.33, 804.19],
                    [-0.07, 38.22, 910.17],
                    [-0.03, 6.21, 981.1],
                    [0, 0, 0],
                ],
                dtype=np.float64,
            )
            * 1e-3
        )  # com repere base pour config q = 0

        # compute cog in link frame
        transforms, _, _ = _get_link_matrix_numba(
            np.array([0, 0, 0, 0, 0, 0], dtype=np.float64),
            a,
            d,
            alpha,
            theta,
            representation.value,
        )
        array_t = transforms.copy()
        array_t = np.append(array_t, [np.identity(4)], axis=0)
        for i in range(cog.shape[0]):
            cog[i] = (np.linalg.inv(array_t[i]) @ np.append(cog[i], 1.0))[:3]

        super().__init__(
            a,
            d,
            alpha,
            theta,
            representation,
            q_min,
            q_max,
            q_point_max,
            torque_max,
            masses,
            cog,
        )
