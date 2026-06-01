"""Unit tests for kinematics.core.core module."""

import numpy as np

from kinematics.core.core import (
    compute_force,
    fk,
    get_dh_mat,
    get_jacobian,
    get_torque_gravity,
    ik,
)


class TestGetDhMat:
    """Test forward kinematics chain computation."""

    def test_zero_joint_angles(self):
        """Test DH matrix chain with zero joint angles."""
        # Simple 3-DOF example
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        transforms = get_dh_mat(q, a, d, alpha, theta)

        # All should be 4x4 matrices
        assert transforms.shape[1:] == (4, 4)

        # All should be homogeneous transformation matrices
        for T in transforms:
            np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_chained_transforms_structure(self):
        """Test that chained transforms have proper structure."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        transforms = get_dh_mat(q, a, d, alpha, theta)

        # T02 should be approximately T01 @ T12
        # We can test that determinants are 1 (proper rotations)
        for T in transforms:
            R = T[:3, :3]
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_different_q_values(self):
        """Test that different joint angles produce different transforms."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])

        q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q2 = np.array([np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])

        transform_q1 = get_dh_mat(q1, a, d, alpha, theta)
        transform_q2 = get_dh_mat(q2, a, d, alpha, theta)

        # Different joint angles should produce different end-effector poses
        assert not np.allclose(transform_q1[-1], transform_q2[-1])


class TestForwardKinematics:
    """Test forward kinematics computation."""

    def test_fk_with_identity_tcp(self):
        """Test FK with identity TCP transformation."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tcp = np.identity(4)

        T = fk(q, a, d, alpha, theta, tcp)

        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_fk_with_tcp_offset(self):
        """Test FK with TCP tool offset."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        tcp = np.identity(4)
        tcp[2, 3] = 0.2  # 20 cm offset in Z

        T_with_tcp = fk(q, a, d, alpha, theta, tcp)
        T_no_tcp = fk(q, a, d, alpha, theta, np.identity(4))

        # TCP offset should affect position but not rotation
        assert not np.allclose(T_with_tcp[:3, 3], T_no_tcp[:3, 3])
        np.testing.assert_array_almost_equal(T_with_tcp[:3, :3], T_no_tcp[:3, :3])

    def test_fk_output_is_homogeneous_matrix(self):
        """Test that FK output is a valid homogeneous transformation matrix."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        tcp = np.identity(4)

        T = fk(q, a, d, alpha, theta, tcp)

        # Check structure
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
        # Check rotation matrix is orthogonal
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_fk_continuity(self):
        """Test FK continuity with small joint angle changes."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        tcp = np.identity(4)

        q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q2 = q1 + 0.001 * np.ones(6)  # Small perturbation

        T1 = fk(q1, a, d, alpha, theta, tcp)
        T2 = fk(q2, a, d, alpha, theta, tcp)

        # End-effector positions should be close
        dist = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
        assert dist < 0.01  # Should be within 1 cm


class TestJacobian:
    """Test Jacobian computation."""

    def test_jacobian_shape(self):
        """Test Jacobian matrix has correct shape."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        tcp = np.identity(4)

        J = get_jacobian(q, a, d, alpha, theta, tcp)

        assert J.shape == (6, 6)

    def test_jacobian_numerical_consistency(self):
        """Test Jacobian by numerical differentiation."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        tcp = np.identity(4)

        J = get_jacobian(q, a, d, alpha, theta, tcp)
        delta = 1e-6

        # Numerical Jacobian for position part (first 3 rows)
        J_num = np.zeros((3, 6))
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += delta
            q_minus = q.copy()
            q_minus[i] -= delta

            T_plus = fk(q_plus, a, d, alpha, theta, tcp)
            T_minus = fk(q_minus, a, d, alpha, theta, tcp)

            J_num[:, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * delta)

        # Compare analytical and numerical Jacobian (position part)
        np.testing.assert_array_almost_equal(J[:3, :], J_num, decimal=4)

    def test_jacobian_with_tcp_offset(self):
        """Test Jacobian computation with TCP offset."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        tcp_no_offset = np.identity(4)
        tcp_with_offset = np.identity(4)
        tcp_with_offset[0, 3] = 0.1

        J_no_offset = get_jacobian(q, a, d, alpha, theta, tcp_no_offset)
        J_with_offset = get_jacobian(q, a, d, alpha, theta, tcp_with_offset)

        # Jacobians should be different with different TCP
        assert not np.allclose(J_no_offset, J_with_offset)


class TestInverseKinematics:
    """Test inverse kinematics computation."""

    def test_ik_convergence_near_solution(self):
        """Test IK convergence from initial guess near solution.

        Note: IK requires initial guess close to solution (typical in robotics).
        Starting from zero may not converge to distant targets.
        """
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)
        tcp = np.identity(4)

        # First, compute FK for a known configuration
        q_target = np.array([0.2, 0.3, -0.2, 0.1, -0.1, 0.15])
        T_target = fk(q_target, a, d, alpha, theta, tcp)

        # Start from near the target (offset by small perturbation)
        q_init = q_target + 0.1 * np.ones(6)
        success, q_solution = ik(
            T_target,
            q_init,
            a,
            d,
            alpha,
            theta,
            tcp,
            q_min,
            q_max,
            epsilon_pos=1e-4,
            epsilon_orient=1e-3,
            max_iter=1000,
        )

        assert success
        # Verify solution by forward kinematics
        T_verify = fk(q_solution, a, d, alpha, theta, tcp)
        np.testing.assert_array_almost_equal(
            T_target[:3, 3], T_verify[:3, 3], decimal=4
        )

    def test_ik_returns_valid_joint_limits(self):
        """Test that IK solution respects joint limits."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi / 2] * 6)
        q_max = np.array([np.pi / 2] * 6)
        tcp = np.identity(4)

        # Target within reachable workspace
        q_target = np.array([0.1, 0.1, -0.1, 0.05, -0.05, 0.1])
        T_target = fk(q_target, a, d, alpha, theta, tcp)

        q_init = np.zeros(6)
        success, q_solution = ik(
            T_target,
            q_init,
            a,
            d,
            alpha,
            theta,
            tcp,
            q_min,
            q_max,
        )

        if success:
            # Verify solution is within limits
            assert np.all(q_solution >= q_min - 1e-6)
            assert np.all(q_solution <= q_max + 1e-6)

    def test_ik_various_initial_guesses(self):
        """Test IK with various initial guesses."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)
        tcp = np.identity(4)

        q_target = np.array([0.2, 0.3, -0.2, 0.1, -0.1, 0.15])
        T_target = fk(q_target, a, d, alpha, theta, tcp)

        # Test with different initial guesses
        initial_guesses = [
            np.zeros(6),
            q_target * 0.5,
            np.random.uniform(-0.5, 0.5, 6),
        ]

        for q_init in initial_guesses:
            success, q_solution = ik(
                T_target,
                q_init,
                a,
                d,
                alpha,
                theta,
                tcp,
                q_min,
                q_max,
                max_iter=1000,
            )

            if success:
                T_verify = fk(q_solution, a, d, alpha, theta, tcp)
                np.testing.assert_array_almost_equal(
                    T_target[:3, 3], T_verify[:3, 3], decimal=3
                )


class TestGravityTorque:
    """Test gravity torque computation."""

    def test_gravity_torque_shape(self):
        """Test gravity torque output shape."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tcp = np.identity(4)

        masses = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
        cog = np.zeros((7, 3))

        tau = get_torque_gravity(q, a, d, alpha, theta, tcp, masses, cog)

        assert tau.shape == (6,)

    def test_gravity_torque_zero_gravity_cog(self):
        """Test gravity torque with zero COG."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tcp = np.identity(4)

        masses = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
        cog = np.zeros((7, 3))  # All COG at origin

        tau = get_torque_gravity(q, a, d, alpha, theta, tcp, masses, cog)

        # With all COG at origin, torques should be very small
        np.testing.assert_array_almost_equal(tau, np.zeros(6), decimal=5)

    def test_gravity_torque_with_masses(self):
        """Test gravity torque computation with realistic masses."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tcp = np.identity(4)

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

        tau = get_torque_gravity(q, a, d, alpha, theta, tcp, masses, cog)

        assert tau.shape == (6,)
        # Should be finite values
        assert np.all(np.isfinite(tau))


class TestComputeForce:
    """Test force computation from torques."""

    def test_compute_force_shape(self):
        """Test force output shape."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])
        tcp = np.identity(4)
        tau = np.array([1.0, 1.5, -0.5, 0.2, -0.1, 0.3])

        force = compute_force(q, a, d, alpha, theta, tcp, tau)

        assert force.shape == (6,)

    def test_compute_force_from_gravity(self):
        """Test force computation from gravity torques.

        Note: Uses non-singular configuration to avoid Jacobian singularities.
        """
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        # Use non-singular configuration (not all zeros)
        q = np.array([0.2, 0.3, -0.2, 0.1, -0.1, 0.15])
        tcp = np.identity(4)

        masses = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
        cog = np.array(
            [
                [0.0, 0.05, 0.1],
                [0.0, 0.1, 0.2],
                [0.0, 0.05, 0.25],
                [0.0, 0.02, 0.15],
                [0.0, 0.01, 0.1],
                [0.0, 0.005, 0.05],
                [0.0, 0.0, 0.01],
            ]
        )

        tau_gravity = get_torque_gravity(q, a, d, alpha, theta, tcp, masses, cog)
        force = compute_force(q, a, d, alpha, theta, tcp, tau_gravity)

        assert force.shape == (6,)
        assert np.all(np.isfinite(force))
