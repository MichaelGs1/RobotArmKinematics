"""
Module containing the base configuration class for robotic systems.

This module provides the `BaseConfig` abstract base class which defines the fundamental
configuration parameters for robotic manipulators, including kinematic and dynamic properties.

Classes:
    BaseConfig: Abstract base class for robotic configuration parameters.
"""

from abc import ABC
from enum import Enum

import numpy as np


class RepresentationType(Enum):
    DH = 0
    DH_KHALIL = 1


class IKSolverMethod(Enum):
    TRANSPOSE = 0
    PSEUDO_INVERSE = 1
    NEWTON_RAPHSON = 2
    DLS = 3


class BaseConfig(ABC):
    """
    Abstract base class for robotic configuration parameters.

    This class defines the fundamental parameters required for robotic manipulators,
    including kinematic and dynamic properties. It serves as a base for more specific
    configuration implementations.

    Attributes:
        a (np.ndarray): Denavit-Hartenberg (Kahlil) a parameters.
        d (np.ndarray): Denavit-Hartenberg (Kahlil) d parameters.
        alpha (np.ndarray): Denavit-Hartenberg (Kahlil) alpha parameters.
        theta (np.ndarray): Denavit-Hartenberg (Kahlil) theta parameters.
        represention_type (RepresentationType) : type of representation DH or DH Khalil
        qmin (np.ndarray): Minimum joint limits.
        qmax (np.ndarray): Maximum joint limits.
        tcp (np.ndarray): Tool Center Point transformation matrix.
        q_point_max (np.ndarray | None): Maximum joint velocity.
        torque_max (np.ndarray | None): Maximum joint torque.
        masses (np.ndarray | None): Masses of the robot links.
        cog (np.ndarray | None): Centers of gravity of the robot links.
    """

    __slots__ = (
        "_parameter_a",
        "_parameter_d",
        "_parameter_alpha",
        "_parameter_theta",
        "_parameter_representation_type",
        "_parameter_qmin",
        "_parameter_qmax",
        "_parameter_tcp",
        "_parameter_q_point_max",
        "_parameter_torque_max",
        "_parameter_masses",
        "_parameter_cog_link",
    )

    def __init__(
        self,
        a: np.ndarray,
        d: np.ndarray,
        alpha: np.ndarray,
        theta: np.ndarray,
        representation_type: RepresentationType,
        qmin: np.ndarray,
        qmax: np.ndarray,
        q_point_max: np.ndarray | None = None,
        torque_max: np.ndarray | None = None,
        masses: np.ndarray | None = None,
        cog: np.ndarray | None = None,
        tcp: np.ndarray = np.identity(4),
    ):
        """
        Initialize the BaseConfig with robot parameters.

        Args:
            a (np.ndarray): Denavit-Hartenberg (Kahlil) a parameters.
            d (np.ndarray): Denavit-Hartenberg (Kahlil) d parameters.
            alpha (np.ndarray): Denavit-Hartenberg (Kahlil) alpha parameters.
            theta (np.ndarray): Denavit-Hartenberg (Kahlil) theta parameters.
            qmin (np.ndarray): Minimum joint limits.
            qmax (np.ndarray): Maximum joint limits.
            q_point_max (np.ndarray | None, optional): Maximum joint velocity. Defaults to None.
            torque_max (np.ndarray | None, optional): Maximum joint torque. Defaults to None.
            masses (np.ndarray | None, optional): Masses of the robot links. Defaults to None.
            cog (np.ndarray | None, optional): Centers of gravity of the robot links. Defaults to None.
            tcp (np.ndarray, optional): Tool Center Point transformation matrix. Defaults to identity matrix.
        """
        assert a.shape[0] == d.shape[0]
        assert a.shape[0] == alpha.shape[0]
        assert a.shape[0] == theta.shape[0]

        assert a.shape[0] == qmin.shape[0]
        assert a.shape[0] == qmax.shape[0]

        self._parameter_a: np.ndarray = a
        self._parameter_d: np.ndarray = d
        self._parameter_alpha: np.ndarray = alpha
        self._parameter_theta: np.ndarray = theta

        self._parameter_representation_type = representation_type

        self._parameter_qmin: np.ndarray = qmin
        self._parameter_qmax: np.ndarray = qmax

        self._parameter_tcp: np.ndarray = np.ascontiguousarray(tcp)

        self._parameter_q_point_max: np.ndarray | None = q_point_max
        self._parameter_torque_max: np.ndarray | None = torque_max

        self._parameter_masses: np.ndarray | None = masses
        self._parameter_cog_link: np.ndarray | None = cog

    @property
    def d(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) d parameters."""
        return self._parameter_d

    @property
    def a(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) a parameters."""
        return self._parameter_a

    @property
    def alpha(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) alpha parameters."""
        return self._parameter_alpha

    @property
    def theta(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) theta parameters."""
        return self._parameter_theta

    @property
    def represention_type(self) -> RepresentationType:
        """Get"""
        return self._parameter_representation_type

    @property
    def tcp(self) -> np.ndarray:
        """Get the Tool Center Point transformation matrix."""
        return self._parameter_tcp

    @tcp.setter
    def tcp(self, tcp: np.ndarray) -> None:
        """Set the Tool Center Point transformation matrix."""
        self._parameter_tcp = np.ascontiguousarray(tcp)

    def set_tool_shape(self, mass: float, cog: np.ndarray) -> None:
        """
        Set the mass and center of gravity for the tool.

        Args:
            mass (float): Mass of the tool.
            cog (np.ndarray): Center of gravity of the tool.

        Raises:
            AssertionError: If masses or cog parameters are not initialized.
        """
        assert self._parameter_masses is not None
        self._parameter_masses[-1] = mass
        assert self._parameter_cog_link is not None
        self._parameter_cog_link[-1] = cog

    @property
    def q_min(self) -> np.ndarray:
        """Get the minimum joint limits."""
        return self._parameter_qmin

    @property
    def q_max(self) -> np.ndarray:
        """Get the maximum joint limits."""
        return self._parameter_qmax

    @property
    def joint_velocity_max(self) -> np.ndarray | None:
        """Get the maximum joint velocity."""
        return self._parameter_q_point_max

    @property
    def torque_max(self) -> np.ndarray | None:
        """Get the maximum joint torque."""
        return self._parameter_torque_max

    @property
    def masses(self) -> np.ndarray | None:
        """Get the masses of the robot links."""
        return self._parameter_masses

    @property
    def link_cog(self) -> np.ndarray | None:
        """Get the centers of gravity of the robot links."""
        return self._parameter_cog_link
