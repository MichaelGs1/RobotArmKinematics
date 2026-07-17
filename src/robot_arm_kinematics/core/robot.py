from typing import cast

import numpy as np

from robot_arm_kinematics.config.config import BaseConfig
from robot_arm_kinematics.core.core import (
    _compute_force_ellipsoid_numba,
    _compute_force_numba,
    _compute_normalize_force_ellipsoid_numba,
    _compute_normalize_velocity_ellipsoid_numba,
    _compute_velocity_ellipsoid_numba,
    _fk_numba,
    _get_amplitude_ellipsoid_numba,
    _get_jacobian_numba,
    _get_link_matrix_numba,
    _get_torque_gravity_numba,
    _ik_numba,
)


class RobotArmKinematics:
    __slots__ = "config"

    def __init__(self, config: BaseConfig):
        self.config = config

    def get_link_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute homogeneous transformation matrices for all n joints using Denavit-Hartenberg parameters.

        Calculates the transformation matrices from frame 0 to each joint frame for a n-DOF robot arm
        using the modified Denavit-Hartenberg (Khalil) convention.

            Args:
                q (np.ndarray): Joint angles (rad), shape (n,).

            Returns:
                np.ndarray: Array of n homogeneous transformation matrices (4x4) from base to each joint frame.
        """
        return cast(
            np.ndarray,
            _get_link_matrix_numba(
                q,
                self.config.a,
                self.config.d,
                self.config.alpha,
                self.config.theta,
                self.config.represention_type.value,
            ),
        )

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the 6xn Jacobian matrix for the robot end-effector.

        Calculates the analytical Jacobian relating joint velocities to end-effector
        linear and angular velocities using the geometric method.
            Args:
                q: Joint angles (rad), shape (n,).
            Returns:
                np.ndarray: 6xn Jacobian matrix (first 3 rows for linear velocity, last 3 for angular).
        """
        return cast(
            np.ndarray,
            _get_jacobian_numba(
                q,
                self.config.a,
                self.config.d,
                self.config.alpha,
                self.config.theta,
                self.config.represention_type.value,
                self.config.tcp,
            ),
        )

    def get_gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques for all n joints.

        Calculates the required joint torques to compensate for gravitational forces
        acting on all robot links and the tool. Uses the center of gravity of each segment.


            Args:
                q (np.ndarray): Joint angles (rad), shape (n,).

            Returns:
                np.ndarray: Gravity compensation torques for all n joints (N.m), shape (n,).
        """
        return cast(
            np.ndarray,
            _get_torque_gravity_numba(
                q,
                self.config.a,
                self.config.d,
                self.config.alpha,
                self.config.theta,
                self.config.represention_type.value,
                self.config.tcp,
                self.config.masses,
                self.config.link_cog,
            ),
        )

    def get_force_tcp(self, q: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Convert joint torques to end-effector forces and moments using the Jacobian transpose.

        Computes the Cartesian forces and moments at the tool from the given joint torques
        using the inverse transpose of the Jacobian matrix.

            Args:
                q (np.ndarray): Joint angles (rad), shape (n,).
                tau (np.ndarray): Joint torques (N.m), shape (n,).

            Returns:
                np.ndarray: End-effector force/moment vector (3 forces + 3 moments), shape (n,).
        """
        return cast(
            np.ndarray,
            _compute_force_numba(
                q,
                self.config.a,
                self.config.d,
                self.config.alpha,
                self.config.theta,
                self.config.represention_type.value,
                self.config.tcp,
                tau,
            ),
        )

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: compute end-effector pose from joint angles.

        Calculates the homogeneous transformation matrix from the base frame to the
        end-effector (tool) frame given the joint configuration.


            Args:
                q (np.ndarray): Joint angles (rad), shape (n,).

            Returns:
                np.ndarray: Homogeneous transformation matrix (4x4) from base to tool frame.
        """
        return cast(
            np.ndarray,
            _fk_numba(
                q,
                self.config.a,
                self.config.d,
                self.config.alpha,
                self.config.theta,
                self.config.represention_type.value,
                self.config.tcp,
            ),
        )

    def ik(
        self,
        target_pose_matrix: np.ndarray,
        q_init: np.ndarray,
        epsilon_pos: float = 1e-4,
        epsilon_orient: float = 1e-3,
        max_iter: int = 1000,
        alpha_fix: float = 0.2,
    ) -> tuple[bool, np.ndarray]:
        """Inverse kinematics: compute joint angles from desired end-effector pose.

        Solves the inverse kinematics problem using iterative numerical method (Newton-Raphson)
        with pseudo-inverse of Jacobian. Minimizes both position and orientation errors.


            Args:
                target_pose_matrix (np.ndarray): Desired end-effector homogeneous transformation (4x4).
                q_init (np.ndarray): Initial joint angle guess (rad), shape (n,).
                epsilon_pos (float, optional): Position error threshold (m). Defaults to 1e-4.
                epsilon_orient (float, optional): Orientation error threshold (rad). Defaults to 1e-3.
                max_iter (int, optional): Maximum iterations. Defaults to 1000.
                alpha_fix (float, optional): Step size damping factor [0, 1]. Defaults to 0.2.

            Returns:
                np.ndarray: Tuple of (success: bool, joint_angles: np.ndarray shape (n,)).
        """
        return cast(
            tuple[bool, np.ndarray],
            _ik_numba(
                target_pose_matrix,
                q_init,
                self.config.a,
                self.config.d,
                self.config.alpha,
                self.config.theta,
                self.config.represention_type.value,
                self.config.tcp,
                self.config.q_min,
                self.config.q_max,
                epsilon_pos,
                epsilon_orient,
                max_iter,
                alpha_fix,
            ),
        )

    def compute_force_ellipsoid(
        self, J: np.ndarray, normalized: bool = False
    ) -> np.ndarray:
        """Compute the force ellipsoid matrix

        Args:
            J (np.ndarray): Jacobian matrix (translation or rotation), shape (3,).
            normalized (bool, optional): Option to normalize ellipsoid by maximum torque allowed. Defaults to False.

        Returns:
            np.ndarray: ellipsoid matrix, shape (3,3)
        """
        if normalized:
            return cast(
                np.ndarray,
                _compute_normalize_force_ellipsoid_numba(J, self.config.torque_max),
            )
        else:
            return cast(np.ndarray, _compute_force_ellipsoid_numba(J))

    def compute_velocity_ellipsoid(
        self, J: np.ndarray, normalized: bool = False
    ) -> np.ndarray:
        """Compute the velocity ellipsoid matrix

        Args:
            J (np.ndarray): Jacobian matrix (translation or rotation), shape (3,).
            normalized (bool, optional): Option to normalize ellipsoid by maximum velocity allowed. Defaults to False.

        Returns:
            np.ndarray: ellipsoid matrix, shape (3,3)
        """
        if normalized:
            return cast(
                np.ndarray,
                _compute_normalize_velocity_ellipsoid_numba(
                    J, self.config.joint_velocity_max
                ),
            )
        else:
            return cast(np.ndarray, _compute_velocity_ellipsoid_numba(J))

    def get_amplitude_ellipsoid(self, A: np.ndarray, dir: np.ndarray) -> float:
        """Compute the amplitude of an ellipsoid along a given direction.

        Calculates the radius of the ellipsoid in the specified direction.
        Used for manipulability and dexterity analysis.

            Args:
                A (np.ndarray): Ellipsoid matrix, shape (3,3).
                dir (np.ndarray): Direction vector, shape (3,), normalized internally.

            Returns:
                float: Radius of the ellipsoid along the given direction.
        """
        return cast(float, _get_amplitude_ellipsoid_numba(A, dir))
