import numpy as np
from numba import njit

from kinematics.utils.utils_compute import matrix_to_rotvect, rotvect_to_matrix


@njit(cache=True)
def compute_velocity(T: np.ndarray, time_tot: float = 5.0) -> np.ndarray:
    """Compute linear and angular velocities along a trajectory.

    Calculates instantaneous linear and angular velocities at each trajectory segment
    by finite differencing between consecutive poses.

    Args:
        T: Trajectory as array of homogeneous transformation matrices,
           shape (n_points, 4, 4).
        time_tot: Total trajectory time (s), default 5.0.

    Returns:
        Velocity array of shape (n_points-1, 2, 3) containing:
            - [:, 0, :]: linear velocity (m/s)
            - [:, 1, :]: angular velocity (rad/s)
    """
    n = T.shape[0]
    speed = np.zeros((n - 1, 2, 3), dtype=np.float64)  # n-1 intervalles

    positions = T[:, :3, 3].astype(np.float64)
    rotations = T[:, :3, :3].astype(np.float64)

    dt = np.float64(time_tot / (n - 1))

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
def compute_linear_trajectory(
    T_base_pose: np.ndarray, d_vector: np.ndarray, distance: float
) -> np.ndarray:
    """Generate a linear trajectory in Cartesian space.

    Creates a trajectory where the end-effector moves in a straight line from the
    initial pose while maintaining constant orientation.

    Args:
        T_base_pose: Starting end-effector homogeneous transformation (4x4).
        d_vector: Unit direction vector (3,) for linear motion.
        distance: Total distance to travel (m).

    Returns:
        Array of 100 homogeneous transformation matrices (100, 4, 4) along trajectory.
    """
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
def compute_circular_trajectory(
    T_base_pose: np.ndarray,
    p_vector: np.ndarray,
    d_vector: np.ndarray,
    theta_max: float,
) -> np.ndarray:
    """Generate a circular trajectory around a rotation axis.

    Creates a circular arc trajectory by rotating the end-effector around an axis
    passing through a specified center point.

    Args:
        T_base_pose: Starting end-effector homogeneous transformation (4x4).
        p_vector: Center point of rotation (3,).
        d_vector: Unit rotation axis direction (3,).
        theta_max: Maximum rotation angle (rad).

    Returns:
        Array of 100 homogeneous transformation matrices (100, 4, 4) along trajectory.
    """
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
