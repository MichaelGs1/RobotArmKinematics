"""Unit tests for kinematics.core module."""

import numpy as np
import pytest

from kinematics.config.config import BaseConfig, RepresentationType
from kinematics.core.robot import RobotArmKinematics


class SimpleRobotConfig(BaseConfig):
    """Simple 6-DOF robot configuration for testing."""

    def __init__(self):
        # Simple 6-DOF robot parameters
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)

        # Optional parameters
        q_point_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        torque_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        masses = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
        cog = np.array(
            [
                [0.01, 0.05, 0.1],
                [0.0, 0.1, 0.2],
                [0.0, 0.05, 0.25],
                [0.0, 0.02, 0.15],
                [0.0, 0.01, 0.1],
                [0.0, 0.005, 0.05],
                [0.0, 0.0, 0.01],
            ]
        )
        tcp = np.identity(4)

        super().__init__(
            a=a,
            d=d,
            alpha=alpha,
            theta=theta,
            representation_type=RepresentationType.DH_KHALIL,
            qmin=q_min,
            qmax=q_max,
            q_point_max=q_point_max,
            torque_max=torque_max,
            masses=masses,
            cog=cog,
            tcp=tcp,
        )


@pytest.fixture
def robot():
    """Create a simple test robot."""
    config = SimpleRobotConfig()
    return RobotArmKinematics(config)


class TestLinkMatrix:
    """Test forward kinematics chain computation."""

    def test_zero_joint_angles(self, robot):
        """Test link matrix chain with zero joint angles."""
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        transforms, _, _ = robot.get_link_matrix(q)

        # All should be 4x4 matrices
        assert transforms.shape[1:] == (4, 4)

        # All should be homogeneous transformation matrices
        for T in transforms:
            np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_chained_transforms_structure(self, robot):
        """Test that chained transforms have proper structure."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        transforms, _, _ = robot.get_link_matrix(q)

        # Check determinants are 1 (proper rotations)
        for T in transforms:
            R = T[:3, :3]
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_different_q_values(self, robot):
        """Test that different joint angles produce different transforms."""
        q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q2 = np.array([np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])

        transform_q1 = robot.get_link_matrix(q1)
        transform_q2 = robot.get_link_matrix(q2)

        # Different joint angles should produce different end-effector poses
        assert not np.allclose(transform_q1[-1], transform_q2[-1])


class TestForwardKinematics:
    """Test forward kinematics computation."""

    def test_fk_with_identity_tcp(self, robot):
        """Test FK with identity TCP transformation."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        T = robot.fk(q)

        # Check structure
        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

        # Check rotation matrix is orthogonal
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

        # Position should be non-zero (robot has DH offsets)
        assert not np.allclose(T[:3, 3], np.zeros(3))

    def test_fk_with_tcp_offset(self, robot):
        """Test FK with TCP tool offset."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        # Test with original TCP
        T_no_tcp = robot.fk(q)

        # Create a new config with TCP offset
        config_with_offset = SimpleRobotConfig()
        tcp_offset = np.identity(4)
        tcp_offset[2, 3] = 0.2  # 20 cm offset in Z
        config_with_offset.tcp = tcp_offset
        robot_with_tcp = RobotArmKinematics(config_with_offset)

        T_with_tcp = robot_with_tcp.fk(q)

        # TCP offset should add exactly 0.2 in the Z direction (in TCP frame)
        # But in world frame, the offset depends on the end-effector orientation
        offset_in_world = T_no_tcp[:3, :3] @ tcp_offset[:3, 3]
        expected_position = T_no_tcp[:3, 3] + offset_in_world
        np.testing.assert_array_almost_equal(T_with_tcp[:3, 3], expected_position)

        # Rotation should not be affected by TCP translation
        np.testing.assert_array_almost_equal(T_with_tcp[:3, :3], T_no_tcp[:3, :3])

    def test_fk_output_is_homogeneous_matrix(self, robot):
        """Test that FK output is a valid homogeneous transformation matrix."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        T = robot.fk(q)

        # Check structure
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
        # Check rotation matrix is orthogonal
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_fk_continuity(self, robot):
        """Test FK continuity with small joint angle changes."""
        q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q2 = q1 + 0.001 * np.ones(6)  # Small perturbation

        T1 = robot.fk(q1)
        T2 = robot.fk(q2)

        # End-effector positions should be close
        dist = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
        assert dist < 0.01  # Should be within 1 cm


class TestJacobian:
    """Test Jacobian computation."""

    def test_jacobian_shape(self, robot):
        """Test Jacobian matrix has correct shape."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        J = robot.get_jacobian(q)

        assert J.shape == (6, 6)

    def test_jacobian_numerical_consistency(self, robot):
        """Test Jacobian by numerical differentiation."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        J = robot.get_jacobian(q)
        delta = 1e-6

        # Numerical Jacobian for position part (first 3 rows)
        J_num = np.zeros((3, 6))
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += delta
            q_minus = q.copy()
            q_minus[i] -= delta

            T_plus = robot.fk(q_plus)
            T_minus = robot.fk(q_minus)

            J_num[:, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * delta)

        # Compare analytical and numerical Jacobian (position part)
        np.testing.assert_array_almost_equal(J[:3, :], J_num, decimal=4)

    def test_jacobian_with_tcp_offset(self, robot):
        """Test Jacobian computation with TCP offset."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        J_no_offset = robot.get_jacobian(q)

        # Create robot with TCP offset
        config_with_offset = SimpleRobotConfig()
        tcp_with_offset = np.identity(4)
        tcp_with_offset[0, 3] = 0.1
        config_with_offset.tcp = tcp_with_offset
        robot_with_offset = RobotArmKinematics(config_with_offset)

        J_with_offset = robot_with_offset.get_jacobian(q)

        # Jacobians should be different with different TCP
        assert not np.allclose(J_no_offset, J_with_offset)


class TestInverseKinematics:
    """Test inverse kinematics computation."""

    def test_ik_convergence_near_solution(self, robot):
        """Test IK convergence from initial guess near solution.

        Note: IK requires initial guess close to solution (typical in robotics).
        Starting from zero may not converge to distant targets.
        """
        # First, compute FK for a known configuration
        q_target = np.array([0.2, 0.3, -0.2, 0.1, -0.1, 0.15])
        T_target = robot.fk(q_target)

        # Start from near the target (offset by small perturbation)
        q_init = q_target + 0.1 * np.ones(6)
        success, q_solution = robot.ik(
            T_target,
            q_init,
            epsilon_pos=1e-4,
            epsilon_orient=1e-3,
            max_iter=1000,
        )

        assert success
        # Verify solution by forward kinematics
        T_verify = robot.fk(q_solution)
        np.testing.assert_array_almost_equal(
            T_target[:3, 3], T_verify[:3, 3], decimal=4
        )

    def test_ik_returns_valid_joint_limits(self, robot):
        """Test that IK solution respects joint limits."""
        # Target within reachable workspace
        q_target = np.array([0.1, 0.1, -0.1, 0.05, -0.05, 0.1])
        T_target = robot.fk(q_target)

        q_init = np.zeros(6)
        success, q_solution = robot.ik(T_target, q_init)

        if success:
            # Verify solution is within limits
            assert np.all(q_solution >= robot.config.q_min - 1e-6)
            assert np.all(q_solution <= robot.config.q_max + 1e-6)

    def test_ik_various_initial_guesses(self, robot):
        """Test IK with various initial guesses."""
        q_target = np.array([0.2, 0.3, -0.2, 0.1, -0.1, 0.15])
        T_target = robot.fk(q_target)

        # Test with different initial guesses
        initial_guesses = [
            np.zeros(6),
            q_target * 0.5,
            np.random.uniform(-0.5, 0.5, 6),
        ]

        for q_init in initial_guesses:
            success, q_solution = robot.ik(
                T_target,
                q_init,
                max_iter=1000,
            )

            if success:
                T_verify = robot.fk(q_solution)
                np.testing.assert_array_almost_equal(
                    T_target[:3, 3], T_verify[:3, 3], decimal=3
                )


class TestGravityTorque:
    """Test gravity torque computation."""

    def test_gravity_torque_shape(self, robot):
        """Test gravity torque output shape."""
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = robot.get_gravity_torque(q)

        assert tau.shape == (6,)

    def test_gravity_torque_zero_gravity_cog(self):
        """Test gravity torque with zero COG."""
        # Create config with zero COG
        config = SimpleRobotConfig()
        config._parameter_cog_link = np.zeros((7, 3))
        robot = RobotArmKinematics(config)

        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = robot.get_gravity_torque(q)

        # With all COG at origin, torques should be very small
        np.testing.assert_array_almost_equal(tau, np.zeros(6), decimal=5)

    def test_gravity_torque_with_masses(self, robot):
        """Test gravity torque computation with realistic masses."""
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = robot.get_gravity_torque(q)

        assert tau.shape == (6,)
        # Should be finite values
        assert np.all(np.isfinite(tau))


class TestComputeForce:
    """Test force computation from torques."""

    def test_compute_force_shape(self, robot):
        """Test force output shape."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        tau = np.array([1.0, 1.5, -0.5, 0.2, -0.1, 0.3])

        force = robot.get_force_tcp(q, tau)

        assert force.shape == (6,)

    def test_compute_force_from_gravity(self, robot):
        """Test force computation from gravity torques.

        Note: Uses non-singular configuration to avoid Jacobian singularities.
        """
        # Use non-singular configuration (not all zeros)
        q = np.array([0.2, 0.3, -0.2, 0.1, -0.1, 0.15])

        tau_gravity = robot.get_gravity_torque(q)
        force = robot.get_force_tcp(q, tau_gravity)

        assert force.shape == (6,)
        assert np.all(np.isfinite(force))


class TestEllipsoid:
    """Test ellipsoid computation."""

    def test_shape_ellipsoid(self, robot):
        """Test ellipsoid matrix shape."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        J = robot.get_jacobian(q)
        J_trans = J[:3, :]

        A = robot.compute_velocity_ellipsoid(J_trans, normalized=False)
        assert A.shape == (3, 3)

        A = robot.compute_velocity_ellipsoid(J_trans, normalized=True)
        assert A.shape == (3, 3)

        A = robot.compute_force_ellipsoid(J_trans, normalized=False)
        assert A.shape == (3, 3)

        A = robot.compute_force_ellipsoid(J_trans, normalized=True)
        assert A.shape == (3, 3)

    def test_shape_amplitude_ellipsoid(self, robot):
        """Test amplitude ellipsoid shape."""
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        J = robot.get_jacobian(q)
        J_rot = J[3:, :]

        A = robot.compute_velocity_ellipsoid(J_rot, normalized=False)

        u = np.array([0, 1, 0], dtype=np.float64)
        amplitude = robot.get_amplitude_ellipsoid(A, u)

        assert isinstance(amplitude, float)
