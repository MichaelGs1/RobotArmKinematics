import numpy as np

from kinematics.config.config import BaseConfig, RepresentationType
from kinematics.config.doosan_m0609 import DoosanM0609Config
from kinematics.config.kuka_iiwa import KukaIiwaConfig
from kinematics.config.ur20 import UR20Config
from kinematics.utils import dh_mat, dh_mat_khalil


def display_config(config: BaseConfig) -> None:
    print("Config :", config.__class__)
    print("DHM parameters :")
    print(config.a)
    print(config.d)
    print(config.alpha)
    print(config.theta)

    print("TCP : ", config.tcp)

    tcp = np.identity(4)
    tcp[2, 3] = 0.2
    config.tcp = tcp

    print("TCP : ", config.tcp)

    if config.represention_type == RepresentationType.DH:
        res = dh_mat(
            config.a,
            config.d,
            config.alpha,
            config.theta,
        )
        print("Matrixes DH : ", res)
    elif config.represention_type == RepresentationType.DH_KHALIL:
        res = dh_mat_khalil(
            config.a,
            config.d,
            config.alpha,
            config.theta,
        )
        print("Matrix DHM : ", res)

    print("Masses : ", config.masses)
    print("Cog : ", config.link_cog)
    print("\n")


def main() -> None:
    # 7 axis
    config_kuka = KukaIiwaConfig()
    display_config(config_kuka)

    # 6 axis DHM
    config_doosan = DoosanM0609Config()
    display_config(config_doosan)

    # 6 axis DH
    config_ur = UR20Config()
    display_config(config_ur)


if __name__ == "__main__":
    main()
