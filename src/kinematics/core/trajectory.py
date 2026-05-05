import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R

from kinematics.utils.utils_compute import rotvect_to_matrix, matrix_to_rotvect

@njit(cache=True)
def compute_velocity(T, time_tot = 5.0):
    n = T.shape[0]
    speed = np.zeros((n - 1, 2, 3), dtype=np.float32)  # n-1 intervalles

    positions = T[:, :3, 3].astype(np.float32)
    rotations = T[:, :3, :3].astype(np.float32)

    dt = np.float32(time_tot / (n - 1))

    for i in range(n - 1):
        # Vitesse linéaire (norme)
        dp = positions[i + 1] - positions[i]
        v = dp / dt
        speed[i, 0] = v

        # Vitesse angulaire (norme)
        R_rel = np.dot(rotations[i + 1], rotations[i].T)  
        omega = matrix_to_rotvect(R_rel)
        omega = omega / dt
        speed[i, 1] = omega

    return speed


@njit(cache=True)
def compute_linear_trajectory(T_base_pose, d_vector, distance):
    N = 100
    Ts = np.zeros((N, 4, 4))
    index = 0

    for s in np.linspace(0, 1, N):
        p0 = T_base_pose[:3, 3]
        p = p0 + s * distance * d_vector

        R_tool = T_base_pose[:3, :3]  # orientation constante
        
        T = np.eye(4)
        T[:3, :3] = R_tool
        T[:3, 3] = p

        Ts[index] = T
        index += 1
        
    return Ts


@njit(cache=True)
def compute_circular_trajectory(T_base_pose, p_vector, d_vector, theta_max):
    N = 100
    Ts = np.zeros((N, 4, 4))
    index = 0

    for s in np.linspace(0, 1, N):
        theta = s * theta_max
        # Rm = R.from_rotvec(theta * d_vector).as_matrix()
        Rm = np.ascontiguousarray(rotvect_to_matrix(theta * d_vector))

        p0 = T_base_pose[:3, 3]
        p = p_vector + Rm @ (p0 - p_vector)

        R0 = np.ascontiguousarray(T_base_pose[:3, :3])
        R_tool = Rm @ R0
            
        T = np.eye(4)
        T[:3, :3] = R_tool
        T[:3, 3] = p

        Ts[index] = T
        index += 1

    return Ts
