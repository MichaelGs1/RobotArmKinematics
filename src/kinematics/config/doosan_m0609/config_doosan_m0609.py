import numpy as np
from kinematics.config.config import BaseConfig
from kinematics.core import get_dh_mat

class DoosanM0609Config(BaseConfig):
    def __init__(self):
        #doosan dh
        d3 = 0.411
        r1 = 0.135
        r2 = 0.00625
        r4 = 0.368
        r6 = 0.121

        d = np.array([0, 0, d3, 0, 0, 0])
        r = np.array([r1, r2, 0, r4, 0, r6])
        alpha = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])
        theta = np.array([0, -np.pi/2, np.pi/2, 0, 0, 0])

        q_min = np.deg2rad(np.array([-360, -95, -135, -360, -135, -330], dtype=np.float64))
        q_max = np.deg2rad(np.array([ 360, 95,  135,  360,  135,  330], dtype=np.float64))

        q_point_max = np.deg2rad(np.array([150, 150, 180, 225, 225, 225], dtype=np.float64))
        torque_max = np.array([160, 160, 90, 45, 45, 45], dtype=np.float64)

        masses = np.array([5.02, 8.04, 3.6, 3.57, 2.83, 1.16, 0], dtype=np.float64)
        cog = np.array([[0.07, 36.23, 131.58],
                        [0.03, 166.3, 339.47],
                        [-0.02, 49.7, 552.87],
                        [0.04, 103.33, 804.19],
                        [-0.07, 38.22, 910.17],
                        [-0.03, 6.21, 981.1],
                        [0, 0, 0]], dtype=np.float64) * 1e-3        # com repere base pour config q = 0

        # compute cog in link frame
        T01, T02, T03, T04, T05, T06 = get_dh_mat([0,0,0,0,0,0], d, r, alpha, theta)
        T = np.array([T01, T02, T03, T04, T05, T06, np.identity(4)])
        for i in range(cog.shape[0]):
            cog[i] = (np.linalg.inv(T[i]) @ np.append(cog[i], 1.0))[:3]

        super().__init__(d, r, alpha, theta, q_min, q_max, q_point_max, torque_max, masses, cog)