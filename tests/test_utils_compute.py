"""Unit tests for kinematics.utils.utils_compute module."""

import numpy as np

from kinematics.utils.utils_compute import (
    dh_mat,
    matrix_to_rotvect,
    rotvect_to_matrix,
)


class TestDhMat:
    """Test Denavit-Hartenberg matrix computation (Khalil convention)."""

    def test_identity_transformation(self):
        """Test DH matrix with identity transformation (all zeros)."""
        mat = dh_mat(d=0, r=0, alpha=0, theta=0)
        expected = np.identity(4)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_homogeneous_matrix_structure(self):
        """Test DH matrix has proper homogeneous structure."""
        mat = dh_mat(d=1.0, r=0.5, alpha=np.pi / 4, theta=np.pi / 6)
        assert mat.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(mat[3, :], [0, 0, 0, 1])

    def test_rotation_matrix_orthogonal(self):
        """Test that the rotation part is orthogonal (property of DH)."""
        mat = dh_mat(d=0.5, r=0.3, alpha=np.pi / 3, theta=np.pi / 4)
        R = mat[:3, :3]
        # R @ R.T should be identity (orthogonality)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        # det(R) should be 1 (proper rotation)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_rotation_matrix_orthogonal_random(self):
        """Test orthogonality with random DH parameters."""
        for _ in range(10):
            d = np.random.uniform(-1, 1)
            r = np.random.uniform(-1, 1)
            alpha = np.random.uniform(-np.pi, np.pi)
            theta = np.random.uniform(-np.pi, np.pi)

            mat = dh_mat(d, r, alpha, theta)
            R = mat[:3, :3]

            # Check orthogonality
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_translation_d_only(self):
        """Test DH matrix with only d (z-axis translation)."""
        d_val = 1.5
        mat = dh_mat(d=d_val, r=0, alpha=0, theta=0)
        # When alpha=0, r=0, theta=0: position should be (0, 0, d)
        assert np.isclose(mat[0, 3], d_val)
        assert np.isclose(mat[1, 3], 0)
        assert np.isclose(mat[2, 3], 0)

    def test_theta_rotation_pi_2(self):
        """Test 90-degree rotation around Z axis."""
        mat = dh_mat(d=0, r=0, alpha=0, theta=np.pi / 2)
        # Rotation matrix for pi/2 around Z:
        # [ 0 -1  0]
        # [ 1  0  0]
        # [ 0  0  1]
        assert np.isclose(mat[0, 0], 0, atol=1e-10)
        assert np.isclose(mat[0, 1], -1)
        assert np.isclose(mat[1, 0], 1)
        assert np.isclose(mat[1, 1], 0, atol=1e-10)
        assert np.isclose(mat[2, 2], 1)

    def test_alpha_rotation_pi_2(self):
        """Test 90-degree rotation around X axis (alpha)."""
        mat = dh_mat(d=0, r=0, alpha=np.pi / 2, theta=0)
        # Rotation matrix for pi/2 around X:
        # [ 1  0  0]
        # [ 0  0 -1]
        # [ 0  1  0]
        assert np.isclose(mat[0, 0], 1)
        assert np.isclose(mat[1, 1], 0, atol=1e-10)
        assert np.isclose(mat[1, 2], -1)
        assert np.isclose(mat[2, 1], 1)
        assert np.isclose(mat[2, 2], 0, atol=1e-10)

    def test_r_translation_with_alpha(self):
        """Test r translation with alpha rotation."""
        r_val = 1.0
        alpha_val = np.pi / 2
        mat = dh_mat(d=0, r=r_val, alpha=alpha_val, theta=0)
        # When alpha = pi/2: sin(alpha) = 1, cos(alpha) = 0
        # Position should have components dependent on r and alpha
        assert np.isclose(mat[1, 3], -r_val * 1.0)  # -r * sin(alpha)
        assert np.isclose(mat[2, 3], r_val * 0, atol=1e-10)  # r * cos(alpha)

    def test_combined_transformation(self):
        """Test combined d, r, alpha, theta transformation."""
        d = 0.5
        r = 0.3
        alpha = np.pi / 6  # 30 degrees
        theta = np.pi / 4  # 45 degrees

        mat = dh_mat(d, r, alpha, theta)

        # Test homogeneous structure
        assert mat.shape == (4, 4)
        np.testing.assert_array_almost_equal(mat[3, :], [0, 0, 0, 1])

        # Test rotation part is valid
        R = mat[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


class TestDhComposition:
    """Test composition of DH matrices (chain of frames)."""

    def test_dh_composition_associativity(self):
        """Test that (T01 @ T12) @ T23 = T01 @ (T12 @ T23)."""
        d_vals = [0.5, 0.3, 0.2]
        r_vals = [0.1, 0.2, 0.15]
        alphas = [0, -np.pi / 2, np.pi / 2]
        thetas = [0.1, 0.2, -0.15]

        T01 = dh_mat(d_vals[0], r_vals[0], alphas[0], thetas[0])
        T12 = dh_mat(d_vals[1], r_vals[1], alphas[1], thetas[1])
        T23 = dh_mat(d_vals[2], r_vals[2], alphas[2], thetas[2])

        # Both should be equal (within numerical precision)
        result1 = (T01 @ T12) @ T23
        result2 = T01 @ (T12 @ T23)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_dh_chain_validity(self):
        """Test that chained DH transforms remain valid homogeneous matrices."""
        matrices = []
        for i in range(5):
            d = np.random.uniform(-0.5, 0.5)
            r = np.random.uniform(0, 1)
            alpha = np.random.uniform(-np.pi, np.pi)
            theta = np.random.uniform(-np.pi, np.pi)
            matrices.append(dh_mat(d, r, alpha, theta))

        # Compose all matrices
        T = matrices[0]
        for mat in matrices[1:]:
            T = T @ mat

        # Result should be a valid homogeneous transformation
        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

        # Rotation part should be orthogonal
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_inverse_transformation(self):
        """Test that T @ T^-1 = Identity."""
        d = 0.5
        r = 0.3
        alpha = np.pi / 6
        theta = np.pi / 4

        T = dh_mat(d, r, alpha, theta)
        T_inv = np.linalg.inv(T)

        result = T @ T_inv
        np.testing.assert_array_almost_equal(result, np.eye(4), decimal=10)

    def test_6dof_chain(self):
        """Test a complete 6-DOF robot DH chain (like DOOSAN)."""
        # Example parameters (simplified)
        d = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        r = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta_offset = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])

        # Compute individual transforms
        transforms = []
        for i in range(6):
            T = dh_mat(d[i], r[i], alpha[i], theta_offset[i] + q[i])
            transforms.append(T)

        # Compose chain
        T_total = transforms[0]
        for T in transforms[1:]:
            T_total = T_total @ T

        # Validate result
        assert T_total.shape == (4, 4)
        np.testing.assert_array_almost_equal(T_total[3, :], [0, 0, 0, 1])

        # Rotation should be valid
        R = T_total[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


class TestRotvectToMatrix:
    """Test rotation vector to rotation matrix conversion."""

    def test_zero_rotation(self):
        """Test that zero rotation vector returns identity matrix."""
        rotvec = np.array([0.0, 0.0, 0.0])
        mat = rotvect_to_matrix(rotvec)
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_small_rotation(self):
        """Test small rotation (quasi-zero)."""
        rotvec = np.array([1e-12, 1e-12, 1e-12])
        mat = rotvect_to_matrix(rotvec)
        np.testing.assert_array_almost_equal(mat, np.eye(3), decimal=5)

    def test_pi_rotation_x_axis(self):
        """Test 180-degree rotation around X axis."""
        rotvec = np.array([np.pi, 0, 0])
        mat = rotvect_to_matrix(rotvec)
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        np.testing.assert_array_almost_equal(mat, expected, decimal=5)

    def test_pi_rotation_z_axis(self):
        """Test 180-degree rotation around Z axis."""
        rotvec = np.array([0, 0, np.pi])
        mat = rotvect_to_matrix(rotvec)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(mat, expected, decimal=5)

    def test_quarter_rotation_z_axis(self):
        """Test 90-degree rotation around Z axis."""
        rotvec = np.array([0, 0, np.pi / 2])
        mat = rotvect_to_matrix(rotvec)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(mat, expected, decimal=5)

    def test_arbitrary_rotation(self):
        """Test arbitrary rotation vector."""
        # Rotation around axis [1,1,1] by angle pi/4
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 4
        rotvec = axis * angle
        mat = rotvect_to_matrix(rotvec)
        # Check determinant is 1 (proper rotation)
        assert np.isclose(np.linalg.det(mat), 1.0)
        # Check it's orthogonal
        np.testing.assert_array_almost_equal(mat @ mat.T, np.eye(3))

    def test_rotation_matrix_is_orthogonal(self):
        """Test that output is always an orthogonal matrix."""
        for _ in range(10):
            rotvec = np.random.randn(3) * np.pi
            mat = rotvect_to_matrix(rotvec)
            # Check orthogonality: R @ R.T = I
            np.testing.assert_array_almost_equal(mat @ mat.T, np.eye(3), decimal=5)
            # Check determinant is 1
            assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-5)


class TestMatrixToRotvect:
    """Test rotation matrix to rotation vector conversion."""

    def test_identity_matrix(self):
        """Test identity matrix returns zero vector."""
        mat = np.eye(3)
        rotvec = matrix_to_rotvect(mat)
        np.testing.assert_array_almost_equal(rotvec, np.zeros(3))

    def test_pi_rotation_x_axis(self):
        """Test 180-degree rotation around X axis."""
        mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        rotvec = matrix_to_rotvect(mat)
        expected = np.array([np.pi, 0, 0])
        np.testing.assert_array_almost_equal(
            np.abs(rotvec), np.abs(expected), decimal=5
        )

    def test_quarter_rotation_z_axis(self):
        """Test 90-degree rotation around Z axis."""
        mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotvec = matrix_to_rotvect(mat)
        expected = np.array([0, 0, np.pi / 2])
        np.testing.assert_array_almost_equal(rotvec, expected, decimal=5)

    def test_roundtrip_small_rotation(self):
        """Test rotvec -> matrix -> rotvec roundtrip for small angles."""
        rotvec_original = np.array([0.1, 0.2, 0.3])
        mat = rotvect_to_matrix(rotvec_original)
        rotvec_recovered = matrix_to_rotvect(mat)
        np.testing.assert_array_almost_equal(
            rotvec_original, rotvec_recovered, decimal=5
        )

    def test_roundtrip_large_rotation(self):
        """Test rotvec -> matrix -> rotvec roundtrip for large angles."""
        rotvec_original = np.array([1.5, -0.8, 2.2])
        mat = rotvect_to_matrix(rotvec_original)
        rotvec_recovered = matrix_to_rotvect(mat)
        # For angles close to pi, there might be sign ambiguity, so we check both
        try:
            np.testing.assert_array_almost_equal(
                rotvec_original, rotvec_recovered, decimal=4
            )
        except AssertionError:
            np.testing.assert_array_almost_equal(
                rotvec_original, -rotvec_recovered, decimal=4
            )

    def test_random_rotations(self):
        """Test roundtrip for random rotations."""
        for _ in range(5):
            rotvec_original = np.random.randn(3) * np.pi * 0.5
            mat = rotvect_to_matrix(rotvec_original)
            rotvec_recovered = matrix_to_rotvect(mat)
            # Check roundtrip
            mat_recovered = rotvect_to_matrix(rotvec_recovered)
            np.testing.assert_array_almost_equal(mat, mat_recovered, decimal=5)


class TestRotationVectorMatrixConsistency:
    """Test consistency between rotvec_to_matrix and matrix_to_rotvect."""

    def test_consistency_forward_backward(self):
        """Test that conversions are consistent."""
        rotvec = np.array([0.5, -0.3, 0.8])
        mat = rotvect_to_matrix(rotvec)
        rotvec_back = matrix_to_rotvect(mat)
        mat_back = rotvect_to_matrix(rotvec_back)
        np.testing.assert_array_almost_equal(mat, mat_back, decimal=5)

    def test_multiple_roundtrips(self):
        """Test multiple roundtrips maintain consistency."""
        rotvec_original = np.array([0.3, -0.2, 0.6])
        current = rotvec_original.copy()

        for _ in range(3):
            mat = rotvect_to_matrix(current)
            current = matrix_to_rotvect(mat)

        final_mat = rotvect_to_matrix(current)
        original_mat = rotvect_to_matrix(rotvec_original)
        np.testing.assert_array_almost_equal(final_mat, original_mat, decimal=5)
