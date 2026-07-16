import numpy as np

from kinematics.config.config import BaseConfig, RepresentationType


class UR20Config(BaseConfig):
    __slots__ = ()

    def __init__(self) -> None:
        a = np.array([0, -0.8618, -0.7277, 0, 0, 0])
        d = np.array([0.236, 0, 0, 0.2019, 0.1594, 0.1548])
        alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
        theta = np.array([0, 0, 0, 0, 0, 0])

        q_min = np.deg2rad(
            np.array([-360, -360, -360, -360, -360, -360], dtype=np.float64)
        )
        q_max = np.deg2rad(np.array([360, 360, 360, 360, 360, 360], dtype=np.float64))

        representation = RepresentationType.DH

        torque_max = np.array([738, 738, 433, 433, 107, 107], dtype=np.float64)
        q_point_max = np.deg2rad(
            np.array([120, 120, 150, 210, 210, 210], dtype=np.float64)
        )

        masses = np.array(
            [
                16.343,  # shoulder
                29.632,  # upper arm
                7.879,  # forearm
                3.054,  # wrist 1
                3.126,  # wrist 2
                0.846,  # wrist 3
                0,  # tcp
            ],
            dtype=np.float64,
        )
        cog = np.array(
            [
                [0.0000, -0.0610, 0.0062],
                [0.5226, 0.0000, 0.2098],
                [0.3234, 0.0000, 0.0604],
                [0.0000, -0.0026, 0.0393],
                [0.0000, 0.0024, 0.0379],
                [0.0000, -0.0003, -0.0318],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )

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
