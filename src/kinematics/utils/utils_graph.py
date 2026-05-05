import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def get_joint_positions(list_transforms):
    # T01, T02, T03, T04, T05, T06 = get_dh_mat_doosan(q[0], q[1], q[2], q[3], q[4], q[5])
    T01, T02, T03, T04, T05, T06 = list_transforms
    # La position de chaque articulation est donnée par T[i][:3, 3]
    joint_positions = [
        np.array([0, 0, 0]),  # Base (origine)
        T01[:3, 3],
        T02[:3, 3],
        T03[:3, 3],
        T04[:3, 3],
        T05[:3, 3],
        T06[:3, 3],  # Extrémité du robot
    ]
    return joint_positions


def create_graph(max_dim=1.2, title="Workspace"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Configurer les labels et le titre
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    plt.gcf().canvas.manager.set_window_title(title)


    # Ajuster les limites de l'axe pour tout afficher
    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])
    ax.set_zlim([-max_dim, max_dim])

    ax.set_box_aspect([1,1,1])

    return ax


def plot_robot_3d(ax, list_transforms):
    joint_positions = get_joint_positions(list_transforms)

    # Tracer les segments du robot
    for i in range(len(joint_positions) - 1):
        ax.plot(
            [joint_positions[i][0], joint_positions[i+1][0]],
            [joint_positions[i][1], joint_positions[i+1][1]],
            [joint_positions[i][2], joint_positions[i+1][2]],
            'bo-', linewidth=3, label='Robot' if i == 0 else ""
        )

    # Annoter chaque articulation
    for i, pos in enumerate(joint_positions):
        ax.text(pos[0], pos[1], pos[2], f'Joint {i}', fontsize=10)


def plot_tcp(ax, T06, scale=0.25):
    position = T06[:3, 3].T
    rot = T06[:3, :3]
    rx = rot[:3, 0].T
    ry = rot[:3, 1].T
    rz = rot[:3, 2].T
    ax.quiver(position[0], position[1], position[2], rx[0]*scale, rx[1]*scale, rx[2]*scale, color='red', linewidth=3, arrow_length_ratio=0.1)
    ax.quiver(position[0], position[1], position[2], ry[0]*scale, ry[1]*scale, ry[2]*scale, color='green', linewidth=3, arrow_length_ratio=0.1)
    ax.quiver(position[0], position[1], position[2], rz[0]*scale, rz[1]*scale, rz[2]*scale, color='blue', linewidth=3, arrow_length_ratio=0.1)


def plot_frame(ax, T, scale=0.05, name=None):
    p = T[:3, 3]
    Rm = T[:3, :3]
    ax.quiver(*p, *(Rm[:, 0] * scale), color="r")
    ax.quiver(*p, *(Rm[:, 1] * scale), color="g")
    ax.quiver(*p, *(Rm[:, 2] * scale), color="b")
    if name:
        ax.text(*p, name)

def plot_ellipsoid(A, T06, ax, color='b', label=None, scale=1):
    """
    Trace un ellipsoïde de vitesse/force en 3D à la position de l'effecteur final.

    Args:
        A : matrice de l'ellipse
        T06 (np.ndarray): Matrice de transformation 4x4 de l'effecteur final.
        ax (matplotlib.axes): Axe 3D pour le tracé.
        color (str): Couleur de l'ellipsoïde.
        label (str): Légende pour la légende.
    """
    A = np.linalg.inv(A)
    # Position de l'effecteur final
    position = T06[:3, 3]

    # Décomposition spectrale
    eigvals, eigvecs = np.linalg.eigh(A)

    # anti reflexion : determinant > 0
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1  # Inverser le troisième vecteur propre

    # print("Valeur propre de A (vitesse maxi) : ", eigvals)

    # Rayons = sqrt(valeurs propres)
    radii = np.sqrt(eigvals.real) * scale
    # radii = radii / max(np.sqrt(eigvals.real))
    # radii = radii * 0.5

    # Paramétrisation sphérique
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotation
    ellipsoid = np.stack((x, y, z), axis=0)
    ellipsoid = np.einsum('ij,jkl->ikl', eigvecs, ellipsoid)

    # Translation vers la position de l'effecteur final
    ellipsoid = ellipsoid + position[:, np.newaxis, np.newaxis]

    # Tracer l'ellipsoïde
    ax.plot_wireframe(
        ellipsoid[0], ellipsoid[1], ellipsoid[2],
        rstride=4, cstride=4, color=color, alpha=0.7, label=label
    )

    # Ajouter un point pour la légende
    if label:
        ax.scatter(position[0], position[1], position[2], color=color, label=label)


def display_vector(ax, position, vect):
    # Position de l'effecteur final
    ax.quiver(position[0], position[1], position[2], vect[0], vect[1], vect[2], color='k', linewidth=3, arrow_length_ratio=0.1)
