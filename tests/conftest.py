"""Pytest configuration and shared fixtures for kinematics tests."""

import numpy as np
import pytest

from kinematics.config.doosan_m0609.config_doosan_m0609 import DoosanM0609Config
from kinematics.config.ur10.config_ur10 import UR10Config
from kinematics.config.ur20.config_ur20 import UR20Config


@pytest.fixture
def doosan_robot():
    """Fixture providing DoosanM0609 robot configuration."""
    return DoosanM0609Config()


@pytest.fixture
def ur10_robot():
    """Fixture providing UR10 robot configuration."""
    return UR10Config()


@pytest.fixture
def ur20_robot():
    """Fixture providing UR20 robot configuration."""
    return UR20Config()


@pytest.fixture
def sample_joint_angles():
    """Fixture providing sample joint angles for testing."""
    return np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.15])


@pytest.fixture
def zero_joint_angles():
    """Fixture providing zero joint angles."""
    return np.zeros(6)


@pytest.fixture
def random_joint_angles():
    """Fixture providing random joint angles within [-pi, pi]."""
    return np.random.uniform(-np.pi, np.pi, 6)


@pytest.fixture
def tcp_identity():
    """Fixture providing identity TCP transformation."""
    return np.identity(4)


@pytest.fixture
def tcp_with_offset():
    """Fixture providing TCP with offset."""
    tcp = np.identity(4)
    tcp[0, 3] = 0.1  # 10 cm offset in X
    tcp[2, 3] = 0.2  # 20 cm offset in Z
    return tcp


@pytest.fixture
def tcp_rotated():
    """Fixture providing TCP with rotation."""
    tcp = np.identity(4)
    # 90-degree rotation around Z axis
    tcp[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tcp[2, 3] = 0.1  # 10 cm offset in Z
    return tcp


@pytest.fixture
def masses_default():
    """Fixture providing default masses for the robot."""
    return np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])


@pytest.fixture
def cog_default():
    """Fixture providing default center of gravity."""
    return np.array(
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


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )
    config.addinivalue_line(
        "markers",
        "kinematics: mark test as related to kinematics calculations",
    )
