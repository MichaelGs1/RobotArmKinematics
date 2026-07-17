import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_arm_kinematics.core import (
    compute_circular_trajectory,
    compute_linear_trajectory,
    compute_velocity,
)
from robot_arm_kinematics.utils import create_graph, display_vector, plot_frame


def main() -> None:
    # input
    # pose
    p0 = np.array([0.21, -0.1, 0.3])
    R0 = R.from_euler("xyz", [0.2, 0.1, 0.0]).as_matrix()
    T_tool_0 = np.eye(4)
    T_tool_0[:3, :3] = R0
    T_tool_0[:3, 3] = p0

    T_tool_0 = np.array(
        [
            [0.42435823, 0.45504466, -0.78285018, -0.60542826],
            [0.62123019, -0.7753049, -0.11390964, 0.25464062],
            [-0.65878156, -0.43799166, -0.61169448, 0.19890513],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # rotation/translation axis
    d_vector = np.array([-0.29793831, 0.44991835, 0.19755058])
    # d_vector = np.array([0,0,1])
    d_vector = d_vector / np.linalg.norm(d_vector)  # normalisation obligatoire
    # p_vector = np.array([0.2, -0.1, 0.3])
    p_vector = np.array([1.4564467, -0.37573658, 0.06133111])

    # task parameters
    theta_max = np.deg2rad(90)  # rotation
    dist_max = 0.5  # distance

    # compute trajectory and velocity
    T_lin = compute_linear_trajectory(T_tool_0, d_vector, dist_max)
    speed = compute_velocity(T_lin)
    print("Translation")
    print("Linear speed :", speed[0, 0])
    print("Angular speed :", speed[0, 1])

    print(T_lin.shape)

    position_lin = T_lin[:, :3, 3]

    T_cir = compute_circular_trajectory(T_tool_0, p_vector, d_vector, theta_max)
    speed = compute_velocity(T_cir)
    print("Rotation")
    print("Linear speed :", speed[0, 0])
    print("Angular speed :", speed[0, 1])

    position_cir = T_cir[:, :3, 3]

    # display
    ax = create_graph(max_dim=0.5)
    ax.plot(
        position_lin[:, 0], position_lin[:, 1], position_lin[:, 2], label="Trajectoire"
    )

    display_vector(ax, p_vector, d_vector)

    plot_frame(ax, np.eye(4), scale=0.1, name="R0")
    plot_frame(ax, T_tool_0, scale=0.08, name="Tool0")
    for i in range(0, 100, 16):
        plot_frame(ax, T_lin[i], scale=0.05)
    plt.show()

    ax = create_graph(max_dim=0.5)
    ax.plot(
        position_cir[:, 0], position_cir[:, 1], position_cir[:, 2], label="Trajectoire"
    )

    display_vector(ax, p_vector, d_vector)

    plot_frame(ax, np.eye(4), scale=0.1, name="R0")
    plot_frame(ax, T_tool_0, scale=0.08, name="Tool0")
    for i in range(0, 100, 16):
        plot_frame(ax, T_cir[i], scale=0.05)
    plt.show()


if __name__ == "__main__":
    main()
