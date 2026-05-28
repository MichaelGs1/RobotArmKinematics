"""Unit tests for kinematics.config module."""

import numpy as np

from kinematics.config.config import BaseConfig
from kinematics.config.doosan_m0609.config_doosan_m0609 import DoosanM0609Config
from kinematics.config.ur10.config_ur10 import UR10Config
from kinematics.config.ur20.config_ur20 import UR20Config


class TestBaseConfig:
    """Test BaseConfig configuration class."""

    def test_base_config_initialization(self):
        """Test BaseConfig initialization with required parameters."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)

        config = BaseConfig(a, d, alpha, theta, q_min, q_max)

        assert config is not None

    def test_base_config_parameters_getter(self):
        """Test BaseConfig parameter getters."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)

        config = BaseConfig(a, d, alpha, theta, q_min, q_max)

        np.testing.assert_array_equal(config.parameter_d, d)
        np.testing.assert_array_equal(config.parameter_a, a)
        np.testing.assert_array_equal(config.parameter_alpha, alpha)
        np.testing.assert_array_equal(config.parameter_theta, theta)
        np.testing.assert_array_equal(config.parameter_qmin, q_min)
        np.testing.assert_array_equal(config.parameter_qmax, q_max)

    def test_base_config_tcp_property(self):
        """Test TCP property getter and setter."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)

        config = BaseConfig(a, d, alpha, theta, q_min, q_max)

        # Default TCP should be identity
        np.testing.assert_array_equal(config.parameter_tcp, np.identity(4))

        # Set custom TCP
        tcp = np.identity(4)
        tcp[0, 3] = 0.1
        tcp[2, 3] = 0.2
        config.parameter_tcp = tcp

        np.testing.assert_array_equal(config.parameter_tcp, tcp)

    def test_base_config_with_optional_parameters(self):
        """Test BaseConfig with optional parameters."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)
        q_point_max = np.deg2rad(np.array([150, 150, 180, 225, 225, 225]))
        torque_max = np.array([160, 160, 90, 45, 45, 45])

        config = BaseConfig(a, d, alpha, theta, q_min, q_max, q_point_max, torque_max)

        np.testing.assert_array_equal(config.parameter_q_point_max, q_point_max)
        np.testing.assert_array_equal(config.parameter_torque_max, torque_max)

    def test_base_config_with_masses_and_cog(self):
        """Test BaseConfig with masses and center of gravity."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)

        masses = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
        cog = np.zeros((7, 3))

        config = BaseConfig(
            a,
            d,
            alpha,
            theta,
            q_min,
            q_max,
            masses=masses,
            cog=cog,
        )

        np.testing.assert_array_equal(config.parameter_masses, masses)
        np.testing.assert_array_equal(config.parameter_cog, cog)

    def test_set_tool_shape(self):
        """Test setting tool shape (mass and COG)."""
        a = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        d = np.array([0.1, 0.2, 0.0, 0.15, 0.0, 0.05])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2])
        theta = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
        q_min = np.array([-np.pi] * 6)
        q_max = np.array([np.pi] * 6)

        masses = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.0])
        cog = np.array(
            [
                [0.01, 0.05, 0.1],
                [0.0, 0.1, 0.2],
                [0.0, 0.05, 0.25],
                [0.0, 0.02, 0.15],
                [0.0, 0.01, 0.1],
                [0.0, 0.005, 0.05],
                [0.0, 0.0, 0.0],
            ]
        )

        config = BaseConfig(
            a,
            d,
            alpha,
            theta,
            q_min,
            q_max,
            masses=masses,
            cog=cog,
        )

        # Set tool shape
        tool_mass = 1.5
        tool_cog = np.array([0.01, 0.02, 0.15])

        config.set_tool_shape(tool_mass, tool_cog)

        assert config.parameter_masses[-1] == tool_mass
        np.testing.assert_array_equal(config.parameter_cog[-1], tool_cog)


class TestDoosanM0609Config:
    """Test DoosanM0609 robot configuration."""

    def test_doosan_config_initialization(self):
        """Test DoosanM0609Config initializes correctly."""
        config = DoosanM0609Config()

        assert config is not None
        assert config.parameter_d.shape == (6,)
        assert config.parameter_a.shape == (6,)
        assert config.parameter_alpha.shape == (6,)
        assert config.parameter_theta.shape == (6,)

    def test_doosan_joint_limits(self):
        """Test DoosanM0609Config joint limits are symmetric and valid."""
        config = DoosanM0609Config()

        q_min = config.parameter_qmin
        q_max = config.parameter_qmax

        assert len(q_min) == 6
        assert len(q_max) == 6
        # All min should be less than max
        assert np.all(q_min < q_max)

    def test_doosan_dh_parameters(self):
        """Test DoosanM0609Config DH parameters are valid."""
        config = DoosanM0609Config()

        d = config.parameter_d
        a = config.parameter_a
        alpha = config.parameter_alpha
        theta = config.parameter_theta

        assert np.all(np.isfinite(d))
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(theta))

    def test_doosan_masses_and_cog(self):
        """Test DoosanM0609Config masses and COG are valid."""
        config = DoosanM0609Config()

        masses = config.parameter_masses
        cog = config.parameter_cog

        assert masses is not None
        assert cog is not None
        assert masses.shape == (7,)
        assert cog.shape == (7, 3)
        # All masses should be positive or zero
        assert np.all(masses >= 0)
        # All COG coordinates should be finite
        assert np.all(np.isfinite(cog))


class TestUR10Config:
    """Test UR10 robot configuration."""

    def test_ur10_config_initialization(self):
        """Test UR10Config initializes correctly."""
        config = UR10Config()

        assert config is not None
        assert config.parameter_d.shape == (6,)
        assert config.parameter_a.shape == (6,)
        assert config.parameter_alpha.shape == (6,)
        assert config.parameter_theta.shape == (6,)

    def test_ur10_joint_limits(self):
        """Test UR10Config joint limits are valid."""
        config = UR10Config()

        q_min = config.parameter_qmin
        q_max = config.parameter_qmax

        assert len(q_min) == 6
        assert len(q_max) == 6
        assert np.all(q_min < q_max)

    def test_ur10_dh_parameters(self):
        """Test UR10Config DH parameters are valid."""
        config = UR10Config()

        d = config.parameter_d
        a = config.parameter_a
        alpha = config.parameter_alpha
        theta = config.parameter_theta

        assert np.all(np.isfinite(d))
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(theta))


class TestUR20Config:
    """Test UR20 robot configuration."""

    def test_ur20_config_initialization(self):
        """Test UR20Config initializes correctly."""
        config = UR20Config()

        assert config is not None
        assert config.parameter_d.shape == (6,)
        assert config.parameter_a.shape == (6,)
        assert config.parameter_alpha.shape == (6,)
        assert config.parameter_theta.shape == (6,)

    def test_ur20_joint_limits(self):
        """Test UR20Config joint limits are valid."""
        config = UR20Config()

        q_min = config.parameter_qmin
        q_max = config.parameter_qmax

        assert len(q_min) == 6
        assert len(q_max) == 6
        assert np.all(q_min < q_max)

    def test_ur20_dh_parameters(self):
        """Test UR20Config DH parameters are valid."""
        config = UR20Config()

        d = config.parameter_d
        a = config.parameter_a
        alpha = config.parameter_alpha
        theta = config.parameter_theta

        assert np.all(np.isfinite(d))
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(theta))


class TestConfigComparison:
    """Test comparisons between different robot configurations."""

    def test_different_configs_have_different_parameters(self):
        """Test that different robots have different DH parameters."""
        doosan = DoosanM0609Config()
        ur10 = UR10Config()
        ur20 = UR20Config()

        # At least some parameters should differ
        assert not np.allclose(doosan.parameter_d, ur10.parameter_d) or not np.allclose(
            doosan.parameter_a, ur10.parameter_a
        )
        assert not np.allclose(ur10.parameter_d, ur20.parameter_d) or not np.allclose(
            ur10.parameter_a, ur20.parameter_a
        )

    def test_all_configs_have_6_dof(self):
        """Test that all robot configurations have 6 degrees of freedom."""
        configs = [
            DoosanM0609Config(),
            UR10Config(),
            UR20Config(),
        ]

        for config in configs:
            assert len(config.parameter_d) == 6
            assert len(config.parameter_a) == 6
            assert len(config.parameter_alpha) == 6
            assert len(config.parameter_theta) == 6
            assert len(config.parameter_qmin) == 6
            assert len(config.parameter_qmax) == 6
