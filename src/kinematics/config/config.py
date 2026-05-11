from abc import ABC, abstractmethod

import numpy as np


class BaseConfig(ABC):
    def __init__(self, d: np.array, r: np.array, alpha: np.array, theta: np.array , qmin: np.array, qmax: np.array, q_point_max: np.array=None, torque_max: np.array=None, masses: np.array=None, cog: np.array=None, tcp: np.array = np.identity(4)):
        self._parameter_d = d
        self._parameter_r = r
        self._parameter_alpha = alpha
        self._parameter_theta = theta
        
        self._parameter_qmin = qmin
        self._parameter_qmax = qmax

        self._parameter_tcp = tcp

        self._parameter_q_point_max = q_point_max
        self._parameter_torque_max = torque_max

        self._masses_parameter = masses
        self._cog_parameter = cog

    @property
    def parameter_d(self) -> np.array:
        return self._parameter_d

    @property
    def parameter_r(self) -> np.array:
        return self._parameter_r
    
    @property
    def parameter_alpha(self) -> np.array:
        return self._parameter_alpha
    
    @property
    def parameter_theta(self) -> np.array:
        return self._parameter_theta
    
    @property
    def parameter_tcp(self) -> np.array:
        return self._parameter_tcp
    
    @parameter_tcp.setter
    def parameter_tcp(self, tcp:np.array):
        self._parameter_tcp = tcp

    def set_tool_shape(self, mass, cog:np.array):
        self._masses_parameter[-1] = mass
        self._cog_parameter[-1] = cog

    @property
    def parameter_qmin(self) -> np.array:
        return self._parameter_qmin

    @property
    def parameter_qmax(self) -> np.array:
        return self._parameter_qmax
    
    @property
    def parameter_q_point_max(self) -> np.array:
        return self._parameter_q_point_max
    
    @property
    def parameter_torque_max(self) -> np.array:
        return self._parameter_torque_max
    
    @property
    def parameter_masses(self) -> np.array:
        return self._masses_parameter

    @property
    def parameter_cog(self) -> np.array:
        return self._cog_parameter