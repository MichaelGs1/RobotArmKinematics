import numpy as np
from numba import njit

from kinematics.utils.utils_compute import dh_mat, matrix_to_rotvect


@njit(cache=True)
def get_dh_mat(
    q: np.ndarray, a: np.ndarray, d: np.ndarray, alpha: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute homogeneous transformation matrices for all 6 joints using Denavit-Hartenberg parameters.

    Calculates the transformation matrices from frame 0 to each joint frame for a 6-DOF robot
    using the modified Denavit-Hartenberg (Khalil) convention.

    Args:
        q: Joint angles (rad), shape (6,).
        a: Denavit-Hartenberg a parameters, shape (6,).
        d: Denavit-Hartenberg d parameters, shape (6,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (6,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (6,).

    Returns:
        Tuple of 6 homogeneous transformation matrices (4x4) from base to each joint frame.
    """
    T01 = dh_mat(a[0], d[0], alpha[0], theta[0] + q[0])
    T12 = dh_mat(a[1], d[1], alpha[1], theta[1] + q[1])
    T23 = dh_mat(a[2], d[2], alpha[2], theta[2] + q[2])
    T34 = dh_mat(a[3], d[3], alpha[3], theta[3] + q[3])
    T45 = dh_mat(a[4], d[4], alpha[4], theta[4] + q[4])
    T56 = dh_mat(a[5], d[5], alpha[5], theta[5] + q[5])

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56

    return T01, T02, T03, T04, T05, T06


@njit(cache=True)
def get_jacobian(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    tcp: np.ndarray,
) -> np.ndarray:
    """Compute the 6x6 Jacobian matrix for the robot end-effector.

    Calculates the analytical Jacobian relating joint velocities to end-effector
    linear and angular velocities using the geometric method.

    Args:
        q: Joint angles (rad), shape (6,).
        a: Denavit-Hartenberg a parameters, shape (6,).
        d: Denavit-Hartenberg d parameters, shape (6,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (6,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (6,).
        tcp: Tool Center Point transformation matrix (4x4).

    Returns:
        6x6 Jacobian matrix (first 3 rows for linear velocity, last 3 for angular).
    """
    T01, T02, T03, T04, T05, T06 = get_dh_mat(q, a, d, alpha, theta)
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
def get_torque_gravity(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    tcp: np.ndarray,
    masses: np.ndarray,
    cog: np.ndarray,
) -> np.ndarray:
    """Compute gravity compensation torques for all 6 joints.

    Calculates the required joint torques to compensate for gravitational forces
    acting on all robot links and the tool. Uses the center of gravity of each segment.

    Args:
        q: Joint angles (rad), shape (6,).
        a: Denavit-Hartenberg a parameters, shape (6,).
        d: Denavit-Hartenberg d parameters, shape (6,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (6,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (6,).
        tcp: Tool Center Point transformation matrix (4x4).
        masses: Mass of each link, shape (6,).
        cog: Center of gravity of each link in local frame, shape (6, 3).

    Returns:
        Gravity compensation torques for all 6 joints (N.m), shape (6,).
    """
    assert masses is not None
    assert cog is not None

    g = 9.81
    g_vec = np.array([0, 0, -g])
    tau = np.zeros(6)
    n = len(q)

    # compute jacobian at center of gravity
    T01, T02, T03, T04, T05, T06 = get_dh_mat(q, a, d, alpha, theta)
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
        z_i = np.ascontiguousarray(T_array[i][:3, 2])  # axe joint i
        p_i = np.ascontiguousarray(T_array[i][:3, 3])
        for k in range(i, n):
            # position du CoM en repère monde
            p_com = (T_array[k] @ np.append(cog[k], 1))[:3]

            r = p_com - p_i
            F_k = masses[k] * g_vec

            cross_prod = np.cross(r, F_k)
            contrib_link = np.dot(cross_prod, z_i)
            tau[i] += contrib_link

    # --- TCP contribution ---
    p_tool_com = (T06 @ np.append(cog[-1], 1))[:3]
    F_tool = masses[-1] * g_vec

    for i in range(n):
        z_i = np.ascontiguousarray(T_array[i][:3, 2])
        p_i = np.ascontiguousarray(T_array[i][:3, 3])

        r = p_tool_com - p_i
        cross_prod = np.cross(r, F_tool)
        contrib_link = np.dot(cross_prod, z_i)
        tau[i] += contrib_link

    # compute torque to compensate gravity torque compute previously
    # tau = -tau

    return tau


@njit(cache=True)
def compute_force(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    tcp: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Convert joint torques to end-effector forces and moments using the Jacobian transpose.

    Computes the Cartesian forces and moments at the tool from the given joint torques
    using the inverse transpose of the Jacobian matrix.

    Args:
        q: Joint angles (rad), shape (6,).
        a: Denavit-Hartenberg a parameters, shape (6,).
        d: Denavit-Hartenberg d parameters, shape (6,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (6,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (6,).
        tcp: Tool Center Point transformation matrix (4x4).
        tau: Joint torques (N.m), shape (6,).

    Returns:
        End-effector force/moment vector (3 forces + 3 moments), shape (6,).
    """
    J = get_jacobian(q, a, d, alpha, theta, tcp)
    result: np.ndarray = np.linalg.inv(J.T) @ np.ascontiguousarray(tau)
    return result


@njit(cache=True)
def fk(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    tcp: np.ndarray,
) -> np.ndarray:
    """Forward kinematics: compute end-effector pose from joint angles.

    Calculates the homogeneous transformation matrix from the base frame to the
    end-effector (tool) frame given the joint configuration.

    Args:
        q: Joint angles (rad), shape (6,).
        a: Denavit-Hartenberg a parameters, shape (6,).
        d: Denavit-Hartenberg d parameters, shape (6,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (6,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (6,).
        tcp: Tool Center Point transformation matrix (4x4).

    Returns:
        Homogeneous transformation matrix (4x4) from base to tool frame.
    """
    _, _, _, _, _, T06 = get_dh_mat(q, a, d, alpha, theta)
    T0tool: np.ndarray = T06 @ tcp
    return T0tool


@njit(cache=True)
def ik(
    target_pose_matrix: np.ndarray,
    q_init: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    tcp: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    epsilon_pos: float = 1e-4,
    epsilon_orient: float = 1e-3,
    max_iter: int = 1000,
    alpha_fix: float = 0.2,
) -> tuple[bool, np.ndarray]:
    """Inverse kinematics: compute joint angles from desired end-effector pose.

    Solves the inverse kinematics problem using iterative numerical method (Newton-Raphson)
    with pseudo-inverse of Jacobian. Minimizes both position and orientation errors.

    Args:
        target_pose_matrix: Desired end-effector homogeneous transformation (4x4).
        q_init: Initial joint angle guess (rad), shape (6,).
        a: Denavit-Hartenberg a parameters, shape (6,).
        d: Denavit-Hartenberg d parameters, shape (6,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (6,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (6,).
        tcp: Tool Center Point transformation matrix (4x4).
        q_min: Minimum joint angles (rad), shape (6,).
        q_max: Maximum joint angles (rad), shape (6,).
        epsilon_pos: Position error threshold (m), default 1e-4.
        epsilon_orient: Orientation error threshold (rad), default 1e-3.
        max_iter: Maximum iterations, default 1000.
        alpha_fix: Step size damping factor [0, 1], default 0.2.

    Returns:
        Tuple of (success: bool, joint_angles: np.ndarray shape (6,)).
    """
    q = q_init.copy()
    pos_target = np.ascontiguousarray(target_pose_matrix[:3, 3].T)
    rot_target = np.ascontiguousarray(target_pose_matrix[:3, :3])
    find_solution = False

    for i in range(max_iter):
        T = fk(q, a, d, alpha, theta, tcp)
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
        if (
            np.linalg.norm(e_pos) < epsilon_pos
            and np.linalg.norm(e_orient) < epsilon_orient
        ):
            find_solution = True
            break

        # --- correction via pseudo-inverse ---
        error = np.hstack((e_pos, e_orient))  # 6x1
        J = get_jacobian(q, a, d, alpha, theta, tcp)  # 6x6

        dq = alpha_fix * np.dot(np.linalg.pinv(J), error)
        q += dq
        q = np.clip(q, q_min, q_max)

    return find_solution, q


@njit(cache=True)
def get_amplitude_ellipsoid(A: np.ndarray, dir: np.ndarray) -> float:
    """Compute the amplitude of an ellipsoid along a given direction.

    Calculates the radius of the ellipsoid in the specified direction.
    Used for manipulability and dexterity analysis.

    Args:
        A: Ellipsoid matrix (3x3).
        dir: Direction vector (3,), normalized internally.

    Returns:
        Radius of the ellipsoid along the given direction (scalar).
    """
    A = A.astype(np.float64)
    dir = dir.astype(np.float64)

    norm = np.linalg.norm(dir)
    dir_norm = dir / norm
    square_root = np.sqrt(dir_norm.T @ A @ dir_norm)
    alpha_max: float = 1.0 / square_root
    return alpha_max
