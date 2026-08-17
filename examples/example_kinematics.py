import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_arm_kinematics.config.config import IKSolverMethod
from robot_arm_kinematics.config.doosan_m0609 import DoosanM0609Config
from robot_arm_kinematics.core import RobotArmKinematics
from robot_arm_kinematics.utils import (
    create_graph,
    display_vector,
    plot_ellipsoid,
    plot_frame,
    plot_robot_3d,
    plot_tcp,
)


def main() -> None:
    config = DoosanM0609Config()
    robot = RobotArmKinematics(config)

    # # tcp pose
    # tcp = np.identity(4)
    # tcp[2, 3] = 0.2759
    # config.tcp = tcp
    # # tool shape
    # config.set_tool_shape(1.280, np.array([-0.01396, 0.0067, 0.195]))

    q = np.deg2rad(np.array([0, 0, -90, 0, -90, 180]))  # base
    # q = np.random.uniform(config.q_min, config.q_max)  # random
    print(q)

    # get robot link
    transforms, _, _ = robot.get_link_matrix(q)

    # test fk
    T06 = robot.fk(q)

    print("Position : ", T06[0:3, 3].T)
    orient_mat = T06[:3, :3]
    rot = R.from_matrix(orient_mat)
    print("Rotation : ", rot.as_euler("ZYZ", degrees=True), "\n")

    # test jacobian
    J = robot.get_jacobian(q)

    eps = 1e-6
    for i in range(J.shape[1]):
        print("Joint ", i + 1)
        dq = np.zeros(J.shape[1])
        dq[i] = eps

        T0 = robot.fk(q)
        T1 = robot.fk(q + dq)

        v_num = (T1[:3, 3] - T0[:3, 3]) / eps
        R_rel = T1[:3, :3] @ T0[:3, :3].T
        omega_num = R.from_matrix(R_rel).as_rotvec() / eps

        v_jac = J[:3, i]
        omega_jac = J[3:, i]

        print(np.allclose(v_num, v_jac, atol=1e-4))
        print(np.allclose(omega_num, omega_jac, atol=1e-4))

    print("\n")

    # test ik
    # position = np.array([0.6, 0.6, 0.4])
    position = np.array([0.2, 0.4, 0.2])
    rot = R.from_euler("ZYZ", [0, 90, 90], degrees=True)
    pose = np.identity(4)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = position.T
    print("Pose : ", pose)

    res, q_target = robot.ik(pose, q, solver_method=IKSolverMethod.DLS)
    print("Solution : ", res)
    print("q (rad) : ", q_target)
    print("q (deg) : ", np.rad2deg(q_target))

    # compute motor torque to compensate gravity
    torques = robot.get_gravity_torque(q)
    print("Torques : ", torques)

    force = robot.get_force_tcp(q, torques)
    print("Cartesian forces : ", force)

    # normalisation
    assert config.joint_velocity_max is not None
    assert config.torque_max is not None
    Jscaled_manip = J @ np.diag(config.joint_velocity_max)
    Jscaled_force = np.linalg.inv(np.diag(config.torque_max)) @ J.T

    # test ellipsoid translation
    ax = create_graph(title="Ellipsoid translation")
    plot_robot_3d(ax, transforms)
    plot_tcp(ax, T06)
    plot_frame(ax, np.identity(4), 0.5)

    # Ellipsoïde de vitesse
    A_v = Jscaled_manip[:3, :] @ Jscaled_manip[:3, :].T

    # Ellipsoïde de force
    A_f = np.linalg.inv(Jscaled_force[:3, :] @ Jscaled_force[:3, :].T)

    plot_ellipsoid(A_v, T06, ax, color="blue", label="Velocity")
    plot_ellipsoid(A_f, T06, ax, color="red", label="Force")

    # TCP vector translational speed
    vect_tcp_v = np.array([0.093, -0.995, 0.007], dtype=np.float64).T
    display_vector(ax, T06[:3, 3].T, vect_tcp_v)

    speed_v_amp = robot.get_amplitude_ellipsoid(A_v, vect_tcp_v)
    force_v_amp = robot.get_amplitude_ellipsoid(A_f, vect_tcp_v)
    print("Speed v amp : ", speed_v_amp)
    print("Force v amp : ", force_v_amp)

    plt.tight_layout()
    plt.show()

    # test ellipsoid rotation
    ax = create_graph(title="Ellipsoid rotation")
    plot_robot_3d(ax, transforms)
    plot_tcp(ax, T06)
    plot_frame(ax, np.identity(4), 0.5)

    # Ellipsoïde de vitesse
    A_v = Jscaled_manip[3:, :] @ Jscaled_manip[3:, :].T

    # Ellipsoïde de force
    A_f = np.linalg.inv(Jscaled_force[3:, :] @ Jscaled_force[3:, :].T)

    plot_ellipsoid(A_v, T06, ax, color="blue", label="Velocity")
    plot_ellipsoid(A_f, T06, ax, color="red", label="Force")

    # TCP vector angular speed
    vect_tcp_omega = np.array([0.093, -0.995, 0.007], dtype=np.float64).T
    display_vector(ax, T06[:3, 3].T, vect_tcp_omega)
    speed_omega_amp = robot.get_amplitude_ellipsoid(A_v, vect_tcp_omega)
    force_omega_amp = robot.get_amplitude_ellipsoid(A_f, vect_tcp_omega)
    print("Speed omega amp : ", speed_omega_amp)
    print("Force omega amp : ", force_omega_amp)

    plt.tight_layout()
    plt.show()


# import cProfile

# cProfile.run("main()")

if __name__ == "__main__":
    main()
