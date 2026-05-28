"""
Module containing the base configuration class for robotic systems.

This module provides the `BaseConfig` abstract base class which defines the fundamental
configuration parameters for robotic manipulators, including kinematic and dynamic properties.

Classes:
    BaseConfig: Abstract base class for robotic configuration parameters.
"""

from abc import ABC

import numpy as np


class BaseConfig(ABC):
    """
    Abstract base class for robotic configuration parameters.

    This class defines the fundamental parameters required for robotic manipulators,
    including kinematic and dynamic properties. It serves as a base for more specific
    configuration implementations.

    Attributes:
        parameter_a (np.ndarray): Denavit-Hartenberg (Kahlil) a parameters.
        parameter_d (np.ndarray): Denavit-Hartenberg (Kahlil) d parameters.
        parameter_alpha (np.ndarray): Denavit-Hartenberg (Kahlil) alpha parameters.
        parameter_theta (np.ndarray): Denavit-Hartenberg (Kahlil) theta parameters.
        parameter_qmin (np.ndarray): Minimum joint limits.
        parameter_qmax (np.ndarray): Maximum joint limits.
        parameter_tcp (np.ndarray): Tool Center Point transformation matrix.
        parameter_q_point_max (np.ndarray | None): Maximum joint velocity.
        parameter_torque_max (np.ndarray | None): Maximum joint torque.
        parameter_masses (np.ndarray | None): Masses of the robot links.
        parameter_cog (np.ndarray | None): Centers of gravity of the robot links.
    """

    __slots__ = (
        "_parameter_a",
        "_parameter_d",
        "_parameter_alpha",
        "_parameter_theta",
        "_parameter_qmin",
        "_parameter_qmax",
        "_parameter_tcp",
        "_parameter_q_point_max",
        "_parameter_torque_max",
        "_masses_parameter",
        "_cog_parameter",
    )

    def __init__(
        self,
        a: np.ndarray,
        d: np.ndarray,
        alpha: np.ndarray,
        theta: np.ndarray,
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
        self._parameter_a: np.ndarray = a
        self._parameter_d: np.ndarray = d
        self._parameter_alpha: np.ndarray = alpha
        self._parameter_theta: np.ndarray = theta

        self._parameter_qmin: np.ndarray = qmin
        self._parameter_qmax: np.ndarray = qmax

        self._parameter_tcp: np.ndarray = tcp

        self._parameter_q_point_max: np.ndarray | None = q_point_max
        self._parameter_torque_max: np.ndarray | None = torque_max

        self._masses_parameter: np.ndarray | None = masses
        self._cog_parameter: np.ndarray | None = cog

    @property
    def parameter_d(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) d parameters."""
        return self._parameter_d

    @property
    def parameter_a(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) a parameters."""
        return self._parameter_a

    @property
    def parameter_alpha(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) alpha parameters."""
        return self._parameter_alpha

    @property
    def parameter_theta(self) -> np.ndarray:
        """Get the Denavit-Hartenberg (Kahlil) theta parameters."""
        return self._parameter_theta

    @property
    def parameter_tcp(self) -> np.ndarray:
        """Get the Tool Center Point transformation matrix."""
        return self._parameter_tcp

    @parameter_tcp.setter
    def parameter_tcp(self, tcp: np.ndarray) -> None:
        """Set the Tool Center Point transformation matrix."""
        self._parameter_tcp = tcp

    def set_tool_shape(self, mass: float, cog: np.ndarray) -> None:
        """
        Set the mass and center of gravity for the tool.

        Args:
            mass (float): Mass of the tool.
            cog (np.ndarray): Center of gravity of the tool.

        Raises:
            AssertionError: If masses or cog parameters are not initialized.
        """
        assert self._masses_parameter is not None
        self._masses_parameter[-1] = mass
        assert self._cog_parameter is not None
        self._cog_parameter[-1] = cog

    @property
    def parameter_qmin(self) -> np.ndarray:
        """Get the minimum joint limits."""
        return self._parameter_qmin

    @property
    def parameter_qmax(self) -> np.ndarray:
        """Get the maximum joint limits."""
        return self._parameter_qmax

    @property
    def parameter_q_point_max(self) -> np.ndarray | None:
        """Get the maximum joint velocity."""
        return self._parameter_q_point_max

    @property
    def parameter_torque_max(self) -> np.ndarray | None:
        """Get the maximum joint torque."""
        return self._parameter_torque_max

    @property
    def parameter_masses(self) -> np.ndarray | None:
        """Get the masses of the robot links."""
        return self._masses_parameter

    @property
    def parameter_cog(self) -> np.ndarray | None:
        """Get the centers of gravity of the robot links."""
        return self._cog_parameter
