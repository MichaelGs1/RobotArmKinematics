from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from kinematics.config.doosan_m0609 import DoosanM0609Config
from kinematics.core import (
    compute_force,
    fk,
    get_amplitude_ellipsoid,
    get_dh_mat,
    get_jacobian,
    get_torque_gravity,
    ik,
)
from kinematics.utils import (
    create_graph,
    display_vector,
    plot_ellipsoid,
    plot_frame,
    plot_robot_3d,
    plot_tcp,
)


def main() -> None:
    config = DoosanM0609Config()
    # config = UR20Config()
    # config = UR10Config()

    # tcp pose
    tcp = np.identity(4)
    tcp[2, 3] = 0.2759
    config.parameter_tcp = tcp
    # tool shape
    config.set_tool_shape(1.280, np.array([-0.01396, 0.0067, 0.195]))

    q = np.deg2rad(np.array([0, 0, -90, 0, -90, 180]))
    list_transforms = get_dh_mat(
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
    )

    def test_ik_consistency_with_fk(q: np.ndarray) -> None:
        # 2. Calculer la pose initiale avec fk
        T06 = fk(
            q,
            config.parameter_a,
            config.parameter_d,
            config.parameter_alpha,
            config.parameter_theta,
            config.parameter_tcp,
        )
        pose_initiale = T06[:3, 3]  # Position (x, y, z)
        orientation_initiale = T06[:3, :3]  # Matrice de rotation

        # 3. Générer des petites variations autour de la pose initiale
        n_tests = 100
        epsilon_pos = 0.1  # mètre

        for _ in range(n_tests):
            # Variation aléatoire de la position
            delta_pos = np.random.uniform(-epsilon_pos, epsilon_pos, size=3)
            pose_test = pose_initiale + delta_pos

            # Construction de la matrice de transformation test
            T_test = np.eye(4)
            T_test[:3, 3] = pose_test
            # T_test[:3, :3] = orientation_test
            T_test[:3, :3] = T06[:3, :3]

            # 4. Appel à ik pour retrouver q
            res, q_target = ik(
                T_test,
                q,
                config.parameter_a,
                config.parameter_d,
                config.parameter_alpha,
                config.parameter_theta,
                config.parameter_tcp,
                config.parameter_qmin,
                config.parameter_qmax,
            )

            # 5. Vérification que ik a trouvé une solution
            assert res, "ik n'a pas trouvé de solution"

            # 6. Calculer T_target = fk(q_target, ...)
            T_target = fk(
                q_target,
                config.parameter_a,
                config.parameter_d,
                config.parameter_alpha,
                config.parameter_theta,
                config.parameter_tcp,
            )

            # 7. Vérifier que T_target est proche de T_test
            np.testing.assert_allclose(
                T_target[:3, 3],
                T_test[:3, 3],
                atol=1e-3,
                err_msg="La position calculée par fk(ik(T_test)) ne correspond pas à T_test",
            )
            np.testing.assert_allclose(
                T_target[:3, :3],
                T_test[:3, :3],
                atol=1e-2,
                err_msg="L'orientation calculée par fk(ik(T_test)) ne correspond pas à T_test",
            )

    test_ik_consistency_with_fk(q)

    torques = get_torque_gravity(
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
        config.parameter_masses,
        config.parameter_cog,
    )
    print("Torques : ", torques)

    # test fk
    T06 = fk(
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
    )
    print("Position : ", T06[0:3, 3].T)
    orient_mat = T06[:3, :3]
    rot = R.from_matrix(orient_mat)
    print("Rotation : ", rot.as_euler("ZYZ", degrees=True), "\n")
    # print("Rotation : ", rot.as_matrix(), "\n")

    # test jacobian
    J = get_jacobian(
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
    )
    print("Jacobian : ", J)

    eps = 1e-6
    for i in range(6):
        print("Joint ", i + 1)
        dq = np.zeros(6)
        dq[i] = eps

        T0 = fk(
            q,
            config.parameter_a,
            config.parameter_d,
            config.parameter_alpha,
            config.parameter_theta,
            config.parameter_tcp,
        )
        T1 = fk(
            q + dq,
            config.parameter_a,
            config.parameter_d,
            config.parameter_alpha,
            config.parameter_theta,
            config.parameter_tcp,
        )

        v_num = (T1[:3, 3] - T0[:3, 3]) / eps
        R_rel = T1[:3, :3] @ T0[:3, :3].T
        omega_num = R.from_matrix(R_rel).as_rotvec() / eps

        v_jac = J[:3, i]
        omega_jac = J[3:, i]

        print(np.allclose(v_num, v_jac, atol=1e-4))
        print(np.allclose(omega_num, omega_jac, atol=1e-4))

    print("\n")

    # test ik
    position = np.array([0.3, 0.4, 0.1])
    rot = R.from_euler("ZYZ", [0, 90, 90], degrees=True)
    pose = np.identity(4)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = position.T
    print("Pose : ", pose)

    pose = np.array(
        [
            [-0.95035382, 0.19074547, 0.2458531, -0.58782699],
            [0.10667218, 0.94192023, -0.31844511, 0.23962798],
            [-0.29231597, -0.27640985, -0.91550476, 0.18440304],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    print(pose.dtype)
    print(q)

    # time 2e compute : 0.009s
    t1 = perf_counter()
    res, q_target = ik(
        pose,
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
        config.parameter_qmin,
        config.parameter_qmax,
    )
    t2 = perf_counter()
    print("Solution : ", res)
    print("Time : ", t2 - t1)
    print("q (rad) : ", q_target)
    print("q (deg) : ", np.rad2deg(q_target))

    position = np.array([-0.3, 0.3, 0.2])
    rot = R.from_euler("ZYZ", [0, 90, 90], degrees=True)
    pose = np.identity(4)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = position.T
    print("Pose : ", pose)

    # time 2e compute : 0.009s
    t1 = perf_counter()
    res, q_target = ik(
        pose,
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
        config.parameter_qmin,
        config.parameter_qmax,
    )
    t2 = perf_counter()
    print("Solution : ", res)
    print("Time : ", t2 - t1)
    print("q (rad) : ", q_target)
    print("q (deg) : ", np.rad2deg(q_target))

    # compute motor torque to compensate gravity
    torques = get_torque_gravity(
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
        config.parameter_masses,
        config.parameter_cog,
    )
    print("Torques : ", torques)
    force = compute_force(
        q,
        config.parameter_a,
        config.parameter_d,
        config.parameter_alpha,
        config.parameter_theta,
        config.parameter_tcp,
        torques,
    )
    print("Cartesian forces : ", force)

    # normalisation
    assert config.parameter_q_point_max is not None
    assert config.parameter_torque_max is not None
    Jscaled_manip = J @ np.diag(config.parameter_q_point_max)
    Jscaled_force = np.linalg.inv(np.diag(config.parameter_torque_max)) @ J.T

    # test ellipsoid translation
    ax = create_graph(title="Ellipsoid translation")
    plot_robot_3d(ax, list_transforms)
    plot_tcp(ax, T06)
    plot_frame(ax, np.identity(4))

    # Ellipsoïde de vitesse
    A_v = Jscaled_manip[:3, :] @ Jscaled_manip[:3, :].T

    # Ellipsoïde de force
    A_f = np.linalg.inv(Jscaled_force[:3, :] @ Jscaled_force[:3, :].T)

    plot_ellipsoid(A_v, T06, ax, color="blue", label="Velocity")
    plot_ellipsoid(A_f, T06, ax, color="red", label="Force")

    # TCP vector translational speed
    vect_tcp_v = np.array([0.093, -0.995, 0.007], dtype=np.float32).T
    display_vector(ax, T06[:3, 3].T, vect_tcp_v)

    speed_v_amp = get_amplitude_ellipsoid(A_v, vect_tcp_v)
    force_v_amp = get_amplitude_ellipsoid(A_f, vect_tcp_v)
    print("Speed v amp : ", speed_v_amp)
    print("Force v amp : ", force_v_amp)

    plt.tight_layout()
    plt.show()

    # test ellipsoid rotation
    ax = create_graph(title="Ellipsoid rotation")
    plot_robot_3d(ax, list_transforms)
    plot_tcp(ax, T06)
    plot_frame(ax, np.identity(4))

    # Ellipsoïde de vitesse
    A_v = Jscaled_manip[3:, :] @ Jscaled_manip[3:, :].T

    # Ellipsoïde de force
    A_f = np.linalg.inv(Jscaled_force[3:, :] @ Jscaled_force[3:, :].T)

    plot_ellipsoid(A_v, T06, ax, color="blue", label="Velocity")
    plot_ellipsoid(A_f, T06, ax, color="red", label="Force")

    # TCP vector angular speed
    vect_tcp_omega = np.array([0.093, -0.995, 0.007], dtype=np.float32).T
    display_vector(ax, T06[:3, 3].T, vect_tcp_omega)
    speed_omega_amp = get_amplitude_ellipsoid(A_v, vect_tcp_omega)
    force_omega_amp = get_amplitude_ellipsoid(A_f, vect_tcp_omega)
    print("Speed omega amp : ", speed_omega_amp)
    print("Force omega amp : ", force_omega_amp)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
