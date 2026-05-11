import numpy as np
from kinematics.config.config import BaseConfig

class UR10Config(BaseConfig):
    def __init__(self):
        r = np.array([0.128, 0, 0, 0.1639, 0.1157, 0.0922])
        d = np.array([0, 0, 0.6127, 0.5716, 0, 0])
        alpha = np.array([0, np.pi/2, 0, 0, -np.pi/2, np.pi/2])
        theta = np.array([0, np.pi, 0, 0, 0, np.pi])

        q_min = np.deg2rad(np.array([-360, -360, -360, -360, -360, -360], dtype=np.float64))
        q_max = np.deg2rad(np.array([ 360, 360,  360,  360,  360,  360], dtype=np.float64))

        super().__init__(d, r, alpha, theta, q_min, q_max)