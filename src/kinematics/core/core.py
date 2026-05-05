import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R

from kinematics.config.config import BaseConfig
from kinematics.utils.utils_compute import dh_mat, matrix_to_rotvect

@njit(cache=True)
def get_dh_mat(q, d, r, alpha, theta):
    T01 = dh_mat(d[0], r[0], alpha[0], theta[0] + q[0])
    T12 = dh_mat(d[1], r[1], alpha[1], theta[1] + q[1])
    T23 = dh_mat(d[2], r[2], alpha[2], theta[2] + q[2])
    T34 = dh_mat(d[3], r[3], alpha[3], theta[3] + q[3])
    T45 = dh_mat(d[4], r[4], alpha[4], theta[4] + q[4])
    T56 = dh_mat(d[5], r[5], alpha[5], theta[5] + q[5])

    T02 = T01@T12
    T03 = T02@T23
    T04 = T03@T34
    T05 = T04@T45
    T06 = T05@T56

    return T01, T02, T03, T04, T05, T06


# voir these Adel
@njit(cache=True)
def get_jacobian(q, d, r, alpha, theta, tcp):
    T01, T02, T03, T04, T05, T06 = get_dh_mat(q, d, r, alpha, theta)
    T0tool = T06 @ tcp

    T_array = np.zeros((6, 4, 4))
    T_array[0, :, :] = T01
    T_array[1, :, :] = T02
    T_array[2, :, :] = T03
    T_array[3, :, :] = T04
    T_array[4, :, :] = T05
    T_array[5, :, :] = T06
    
    J = np.zeros((6, 6))
    for i in range(6):
        T = T_array[i]
        z = T[:3, 2]
        o = T[:3, 3]
        v = np.cross(z, T0tool[:3, 3] - o)
        J[:3, i] = v
        J[3:, i] = z

    return J

@njit(cache=True)
def get_torque_gravity(q, d, r, alpha, theta, tcp, masses, cog):
    assert(masses is not None)
    assert(cog is not None)

    g = 9.81
    g_vec = np.array([0, 0, -g])
    tau = np.zeros(6)
    n= len(q)

    # compute jacobian at center of gravity
    T01, T02, T03, T04, T05, T06 = get_dh_mat(q, d, r, alpha, theta)
    T0tool = T06 @ tcp

    T_array = np.zeros((6, 4, 4))
    T_array[0, :, :] = T01
    T_array[1, :, :] = T02
    T_array[2, :, :] = T03
    T_array[3, :, :] = T04
    T_array[4, :, :] = T05
    T_array[5, :, :] = T06

    # --- Segment contribution ---
    n = 6
    for i in range(n):
        z_i = np.ascontiguousarray(T_array[i][:3,2])  # axe joint i
        p_i = np.ascontiguousarray(T_array[i][:3,3])
        for k in range(i, n):
            # position du CoM en repère monde
            p_com = (T_array[k] @ np.append(cog[k],1))[:3]

            r = p_com - p_i
            F_k = masses[k] * g_vec

            cross_prod = np.cross(r, F_k)
            contrib_link = np.dot(cross_prod, z_i)
            tau[i] += contrib_link

    # --- TCP contribution ---
    p_tool_com = (T06 @ np.append(cog[-1],1))[:3]
    F_tool = masses[-1] * g_vec

    for i in range(n):
        z_i = np.ascontiguousarray(T_array[i][:3,2])
        p_i = np.ascontiguousarray(T_array[i][:3,3])

        r = p_tool_com - p_i
        cross_prod = np.cross(r, F_tool)
        contrib_link = np.dot(cross_prod, z_i)
        tau[i] += contrib_link

    # compute torque to compensate gravity torque compute previously
    # tau = -tau

    return tau

@njit(cache=True)
def compute_force(q, d, r, alpha, theta, tcp, tau):
    J = get_jacobian(q, d, r, alpha, theta, tcp)
    return np.linalg.inv(J.T) @ np.ascontiguousarray(tau) 


@njit(cache=True)
def fk(q, d, r, alpha, theta, tcp):
    _, _, _, _, _, T06 = get_dh_mat(q, d, r, alpha, theta)
    T0tool = T06 @ tcp
    return T0tool


@njit(cache=True)
def ik(target_pose_matrix, q_init, d, r, alpha, theta, tcp, q_min, q_max, epsilon_pos=1e-4, epsilon_orient=1e-3, max_iter=1000, alpha_fix=0.2):
    """
    IK avec position + orientation.
    
    target_pose : homegenous transform
    q_init : angles joints initiaux (rad)
    """
    q = q_init.copy()
    pos_target = np.ascontiguousarray(target_pose_matrix[:3, 3].T)
    rot_target = np.ascontiguousarray(target_pose_matrix[:3, :3])
    find_solution = False

    for i in range(max_iter):
        T = fk(q, d, r, alpha, theta, tcp)
        pos_current = np.ascontiguousarray(T[:3, 3])
        rot_current = np.ascontiguousarray(T[:3, :3])
        # print(T)
        
        # --- erreur position ---
        e_pos = pos_target - pos_current
        
        # --- erreur orientation ---
        R_err = np.dot(rot_target, np.linalg.inv(rot_current))
        e_orient = matrix_to_rotvect(R_err)  # en radians
        # e_orient = R.from_matrix(R_err).as_rotvec()
        
        # critère d'arrêt
        if np.linalg.norm(e_pos) < epsilon_pos and np.linalg.norm(e_orient) < epsilon_orient:
            find_solution = True
            break
        
        # --- correction via pseudo-inverse ---
        error = np.hstack((e_pos, e_orient))  # 6x1
        J = get_jacobian(q, d, r, alpha, theta, tcp)  # 6x6

        dq = alpha_fix * np.dot(np.linalg.pinv(J), error)
        q += dq
        q = np.clip(q, q_min, q_max)
        
    return find_solution, q

@njit(cache=True)
def get_amplitude_ellipsoid(A, dir):
    """
    A : np.array(3,3)
    v_dir : np.array(3, 1)
    """
    A = A.astype(np.float64)
    dir = dir.astype(np.float64)
    
    norm = np.linalg.norm(dir) 
    dir_norm = dir / norm
    square_root = np.sqrt(dir_norm.T @ A @ dir_norm)
    alpha_max = 1.0 / square_root
    return alpha_max