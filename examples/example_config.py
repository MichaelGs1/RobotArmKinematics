import numpy as np

from kinematics.config.ur20 import UR20Config
from kinematics.utils import dh_mat


def main() -> None:
    config = UR20Config()
    # config = UR10Config()
    # config = DoosanM0609Config()
    print("DHM parameters :")
    print(config.parameter_d)
    print(config.parameter_r)
    print(config.parameter_alpha)
    print(config.parameter_theta)

    print("TCP : ", config.parameter_tcp)

    tcp = np.identity(4)
    tcp[2, 3] = 0.2
    config.parameter_tcp = tcp

    print("TCP : ", config.parameter_tcp)

    res = dh_mat(
        config.parameter_d[0],
        config.parameter_r[0],
        config.parameter_alpha[0],
        config.parameter_theta[0],
    )
    print("Matrix DHM T01 : ", res)

    print("Masses : ", config.parameter_masses)
    print("Cog : ", config.parameter_cog)


if __name__ == "__main__":
    main()
