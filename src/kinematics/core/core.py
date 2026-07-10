import numpy as np
from numba import float64, int32, njit

from kinematics.config.config import RepresentationType
from kinematics.utils.utils_compute import dh_mat, dh_mat_khalil, matrix_to_rotvect


@njit((float64[:], float64[:], float64[:], float64[:], float64[:], int32), cache=True)
def _get_link_matrix_numba(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    representation_type_value: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute homogeneous transformation matrices for all n joints using Denavit-Hartenberg parameters.

    Calculates the transformation matrices from frame 0 to each joint frame for a n-DOF robot arm
    using the modified Denavit-Hartenberg (Khalil) convention.

    Args:
        q: Joint angles (rad), shape (n,).
        a: Denavit-Hartenberg a parameters, shape (n,).
        d: Denavit-Hartenberg d parameters, shape (n,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (n,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (n,).
        representation_type_value (int): frame represention (DH, ...).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Array of n homogeneous transformation matrices (4x4) from base to each joint frame, origins of link, axes of link
    """
    assert a.shape[0] == q.shape[0]
    n = q.shape[0]

    origins = np.empty((n, 3), dtype=np.float64)
    axes = np.empty((n, 3), dtype=np.float64)

    if representation_type_value == RepresentationType.DH.value:
        # get dh matrixes
        matrixes = dh_mat(a, d, alpha, theta + q)

        origins[0] = np.zeros(3)
        axes[0] = np.array([0.0, 0.0, 1.0])

        transforms_0_n = np.zeros((q.shape[0], 4, 4), dtype=np.float64)
        transforms_0_n[0] = matrixes[0]
        for i in range(1, n):
            # compute dh frame
            transforms_0_n[i] = transforms_0_n[i - 1] @ matrixes[i]

            # get origins and axis of joint
            origins[i] = transforms_0_n[i - 1][:3, 3]
            axes[i] = transforms_0_n[i - 1][:3, 2]

    else:
        # get dh khalil matrixes
        matrixes = dh_mat_khalil(a, d, alpha, theta + q)

        transforms_0_n = np.zeros((q.shape[0], 4, 4), dtype=np.float64)
        transforms_0_n[0] = matrixes[0]
        origins[0] = transforms_0_n[0][:3, 3]
        axes[0] = transforms_0_n[0][:3, 2]
        for i in range(1, n):
            # compute dh khalil frame
            transforms_0_n[i] = transforms_0_n[i - 1] @ matrixes[i]

            # get origins and axis of joint
            origins[i] = transforms_0_n[i][:3, 3]
            axes[i] = transforms_0_n[i][:3, 2]

    return transforms_0_n, origins, axes


@njit(
    float64[:, :](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int32,
        float64[:, ::1],
    ),
    cache=True,
)
def _get_jacobian_numba(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    representation_type_value: int,
    tcp: np.ndarray,
) -> np.ndarray:
    """Compute the 6xn Jacobian matrix for the robot end-effector.

    Calculates the analytical Jacobian relating joint velocities to end-effector
    linear and angular velocities using the geometric method.

    Args:
        q: Joint angles (rad), shape (n,).
        a: Denavit-Hartenberg a parameters, shape (n,).
        d: Denavit-Hartenberg d parameters, shape (n,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (n,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (n,).
        representation_type_value (int): frame represention (DH, ...).
        tcp: Tool Center Point transformation matrix (4x4).

    Returns:
        6xn Jacobian matrix (first 3 rows for linear velocity, last 3 for angular).
    """
    T_array, origins, axes = _get_link_matrix_numba(
        q, a, d, alpha, theta, representation_type_value
    )
    T0tool = np.ascontiguousarray(T_array[-1]) @ tcp

    J = np.zeros((6, T_array.shape[0]))
    for i in range(T_array.shape[0]):
        T = T_array[i]
        z = axes[i]
        o = origins[i]
        v = np.cross(z, T0tool[:3, 3] - o)
        J[:3, i] = v
        J[3:, i] = z

    return J


@njit(
    float64[:](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int32,
        float64[:, ::1],
        float64[:],
        float64[:, :],
    ),
    cache=True,
)
def _get_torque_gravity_numba(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    representation_type_value: int,
    tcp: np.ndarray,
    masses: np.ndarray,
    cog: np.ndarray,
) -> np.ndarray:
    """Compute gravity compensation torques for all n joints.

    Calculates the required joint torques to compensate for gravitational forces
    acting on all robot links and the tool. Uses the center of gravity of each segment.

    Args:
        q: Joint angles (rad), shape (n,).
        a: Denavit-Hartenberg a parameters, shape (n,).
        d: Denavit-Hartenberg d parameters, shape (n,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (n,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (n,).
        representation_type_value (int): frame represention (DH, ...).
        tcp: Tool Center Point transformation matrix (4x4).
        masses: Mass of each link, shape (n,).
        cog: Center of gravity of each link in local frame, shape (n, 3).

    Returns:
        Gravity compensation torques for all n joints (N.m), shape (n,).
    """
    assert masses is not None
    assert cog is not None

    n = q.shape[0]
    g = 9.81
    g_vec = np.array([0, 0, -g])
    tau = np.zeros(n)

    # compute jacobian at center of gravity
    T_array, origins, axes = _get_link_matrix_numba(
        q, a, d, alpha, theta, representation_type_value
    )
    T0tool = T_array[-1] @ tcp

    # --- Segment contribution ---
    for i in range(n):
        z_i = np.ascontiguousarray(axes[i])  # axe joint i
        p_i = np.ascontiguousarray(origins[i])
        for k in range(i, n):
            # position du CoM en repère monde
            p_com = (T_array[k] @ np.append(cog[k], 1))[:3]

            r = p_com - p_i
            F_k = masses[k] * g_vec

            cross_prod = np.cross(r, F_k)
            contrib_link = np.dot(cross_prod, z_i)
            tau[i] += contrib_link

    # --- TCP contribution ---
    p_tool_com = (T_array[-1] @ np.append(cog[-1], 1))[:3]
    F_tool = masses[-1] * g_vec

    for i in range(n):
        z_i = np.ascontiguousarray(axes[i])
        p_i = np.ascontiguousarray(origins[i])

        r = p_tool_com - p_i
        cross_prod = np.cross(r, F_tool)
        contrib_link = np.dot(cross_prod, z_i)
        tau[i] += contrib_link

    return tau


@njit(
    float64[:](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int32,
        float64[:, ::1],
        float64[:],
    ),
    cache=True,
)
def _compute_force_numba(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    representation_type_value: int,
    tcp: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Convert joint torques to end-effector forces and moments using the Jacobian transpose.

    Computes the Cartesian forces and moments at the tool from the given joint torques
    using the inverse transpose of the Jacobian matrix.

    Args:
        q: Joint angles (rad), shape (n,).
        a: Denavit-Hartenberg a parameters, shape (n,).
        d: Denavit-Hartenberg d parameters, shape (n,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (n,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (n,).
        representation_type_value (int): frame represention (DH, ...).
        tcp: Tool Center Point transformation matrix (4x4).
        tau: Joint torques (N.m), shape (n,).

    Returns:
        End-effector force/moment vector (3 forces + 3 moments), shape (n,).
    """
    J = _get_jacobian_numba(q, a, d, alpha, theta, representation_type_value, tcp)
    result: np.ndarray = np.linalg.inv(J.T) @ np.ascontiguousarray(tau)
    return result


@njit(
    float64[:, :](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int32,
        float64[:, ::1],
    ),
    cache=True,
)
def _fk_numba(
    q: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    representation_type_value: int,
    tcp: np.ndarray,
) -> np.ndarray:
    """Forward kinematics: compute end-effector pose from joint angles.

    Calculates the homogeneous transformation matrix from the base frame to the
    end-effector (tool) frame given the joint configuration.

    Args:
        q: Joint angles (rad), shape (n,).
        a: Denavit-Hartenberg a parameters, shape (n,).
        d: Denavit-Hartenberg d parameters, shape (n,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (n,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (n,).
        representation_type_value (int): frame represention (DH, ...).
        tcp: Tool Center Point transformation matrix (4x4).

    Returns:
        Homogeneous transformation matrix (4x4) from base to tool frame.
    """
    transforms, _, _ = _get_link_matrix_numba(
        q, a, d, alpha, theta, representation_type_value
    )
    T0tool: np.ndarray = np.ascontiguousarray(transforms[-1]) @ tcp
    return T0tool


@njit(
    (
        float64[:, :],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int32,
        float64[:, ::1],
        float64[:],
        float64[:],
        float64,
        float64,
        int32,
        float64,
    ),
    cache=True,
)
def _ik_numba(
    target_pose_matrix: np.ndarray,
    q_init: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    representation_type_value: int,
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
        q_init: Initial joint angle guess (rad), shape (n,).
        a: Denavit-Hartenberg a parameters, shape (n,).
        d: Denavit-Hartenberg d parameters, shape (n,).
        alpha: Denavit-Hartenberg alpha parameters (rad), shape (n,).
        theta: Denavit-Hartenberg theta offset parameters (rad), shape (n,).
        representation_type_value (int): frame represention (DH, ...).
        tcp: Tool Center Point transformation matrix (4x4).
        q_min: Minimum joint angles (rad), shape (n,).
        q_max: Maximum joint angles (rad), shape (n,).
        epsilon_pos: Position error threshold (m), default 1e-4.
        epsilon_orient: Orientation error threshold (rad), default 1e-3.
        max_iter: Maximum iterations, default 1000.
        alpha_fix: Step size damping factor [0, 1], default 0.2.

    Returns:
        Tuple of (success: bool, joint_angles: np.ndarray shape (n,)).
    """
    q = q_init.copy()
    pos_target = np.ascontiguousarray(target_pose_matrix[:3, 3].T)
    rot_target = np.ascontiguousarray(target_pose_matrix[:3, :3])
    find_solution = False

    for i in range(max_iter):
        T = _fk_numba(q, a, d, alpha, theta, representation_type_value, tcp)
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
        J = _get_jacobian_numba(
            q, a, d, alpha, theta, representation_type_value, tcp
        )  # 6x6

        dq = alpha_fix * np.dot(np.linalg.pinv(J), error)
        q += dq
        q = np.clip(q, q_min, q_max)

    return find_solution, q


@njit(
    float64[:, :](
        float64[:, ::1],
    ),
    cache=True,
)
def _compute_force_ellipsoid_numba(J: np.ndarray) -> np.ndarray:
    """Compute the force ellipsoid matrix

    Args:
        J (np.ndarray): jacobian matrix (3xn)

    Returns:
        np.ndarray: ellipsoid matrix (3x3)
    """
    Af: np.ndarray = J @ J.T
    return Af


@njit(float64[:, :](float64[:, ::1], float64[:]), cache=True)
def _compute_normalize_force_ellipsoid_numba(
    J: np.ndarray, torque_max: np.ndarray
) -> np.ndarray:
    """Compute the force ellipsoid matrix normalized by torque max of the robot

    Args:
        J (np.ndarray): jacobian matrix (3xn)
        torque_max (np.ndarray): vector of torque max

    Returns:
        np.ndarray: ellipsoid matrix (3x3)
    """
    Df_inv2 = np.diag(1 / torque_max**2)
    Af: np.ndarray = J @ Df_inv2 @ J.T
    return Af


@njit(
    float64[:, :](
        float64[:, ::1],
    ),
    cache=True,
)
def _compute_velocity_ellipsoid_numba(J: np.ndarray) -> np.ndarray:
    """Compute the velocity ellipsoid matrix

    Args:
        J (np.ndarray): jacobian matrix (3xn)

    Returns:
        np.ndarray: ellipsoid matrix (3x3)
    """
    Av = np.linalg.inv(J @ J.T)
    return Av


@njit(float64[:, :](float64[:, ::1], float64[:]), cache=True)
def _compute_normalize_velocity_ellipsoid_numba(
    J: np.ndarray, velocity_max: np.ndarray
) -> np.ndarray:
    """_summary_

    Args:
        J (np.ndarray): jacobian matrix (3xn)
        velocity_max (np.ndarray):  vector of velocity max

    Returns:
        np.ndarray: ellipsoid matrix (3x3)
    """
    Dq = np.diag(velocity_max)
    Av = np.linalg.inv(J @ Dq**2 @ J.T)
    return Av


@njit(float64(float64[:, :], float64[:]), cache=True)
def _get_amplitude_ellipsoid_numba(A: np.ndarray, dir: np.ndarray) -> float:
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
