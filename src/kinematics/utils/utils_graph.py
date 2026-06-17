from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def get_joint_positions(transforms: np.ndarray) -> np.ndarray:
    """Extract 3D positions of all robot joints from transformation matrices.

    Args:
        transforms: Array of n homogeneous transformation matrices (4x4),
                        one for each joint frame.

    Returns:
        Array of n+1 3D position vectors: base origin plus positions of all n joints.
    """
    joint_positions = np.zeros((transforms.shape[0] + 1, 3))  # i : 0 => Base(0,0,0)
    for i in range(transforms.shape[0]):
        joint_positions[i + 1] = transforms[i][:3, 3]

    return joint_positions


def create_graph(max_dim: float = 1.2, title: str = "Workspace") -> Any:
    """Create a 3D matplotlib figure for robot visualization.

    Args:
        max_dim: Half-width of the cubic workspace visualization (m), default 1.2.
        title: Title for the plot window and axes, default "Workspace".

    Returns:
        matplotlib 3D axis object for plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax: Any = fig.add_subplot(111, projection="3d")
    # Configurer les labels et le titre
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    fig = plt.gcf()
    manager = fig.canvas.manager

    if manager is not None:
        manager.set_window_title(title)

    # Ajuster les limites de l'axe pour tout afficher
    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])
    ax.set_zlim([-max_dim, max_dim])

    ax.set_box_aspect([1, 1, 1])

    return ax


def plot_robot_3d(ax: Any, transforms: np.ndarray) -> None:
    """Plot the robot skeleton as connected line segments with joint markers.

    Args:
        ax: matplotlib 3D axis object.
        transforms: List of n homogeneous transformation matrices (4x4).
    """
    joint_positions = get_joint_positions(transforms)

    # Tracer les segments du robot
    for i in range(len(joint_positions) - 1):
        ax.plot(
            [joint_positions[i][0], joint_positions[i + 1][0]],
            [joint_positions[i][1], joint_positions[i + 1][1]],
            [joint_positions[i][2], joint_positions[i + 1][2]],
            "bo-",
            linewidth=3,
            label="Robot" if i == 0 else "",
        )

    # Annoter chaque articulation
    for i, pos in enumerate(joint_positions):
        ax.text(pos[0], pos[1], pos[2], f"Joint {i}", fontsize=10)


def plot_tcp(ax: Any, t_end: np.ndarray, scale: float = 0.25) -> None:
    """Plot the tool center point reference frame with RGB axes.

    Displays red (X), green (Y), and blue (Z) arrows representing the end-effector
    orientation axes at the given pose.

    Args:
        ax: matplotlib 3D axis object.
        t_end: End-effector homogeneous transformation matrix (4x4).
        scale: Arrow length scale factor, default 0.25.
    """
    position = t_end[:3, 3].T
    rot = t_end[:3, :3]
    rx = rot[:3, 0].T
    ry = rot[:3, 1].T
    rz = rot[:3, 2].T
    ax.quiver(
        position[0],
        position[1],
        position[2],
        rx[0] * scale,
        rx[1] * scale,
        rx[2] * scale,
        color="red",
        linewidth=3,
        arrow_length_ratio=0.1,
    )
    ax.quiver(
        position[0],
        position[1],
        position[2],
        ry[0] * scale,
        ry[1] * scale,
        ry[2] * scale,
        color="green",
        linewidth=3,
        arrow_length_ratio=0.1,
    )
    ax.quiver(
        position[0],
        position[1],
        position[2],
        rz[0] * scale,
        rz[1] * scale,
        rz[2] * scale,
        color="blue",
        linewidth=3,
        arrow_length_ratio=0.1,
    )


def plot_frame(
    ax: Any, t: np.ndarray, scale: float = 0.05, name: str | None = None
) -> None:
    """Plot a reference frame with RGB axes at a given transformation.

    Args:
        ax: matplotlib 3D axis object.
        t: Homogeneous transformation matrix (4x4).
        scale: Arrow length scale factor, default 0.05.
        name: Optional label text for the frame.
    """
    p = t[:3, 3]
    rot_mat = t[:3, :3]
    ax.quiver(*p, *(rot_mat[:, 0] * scale), color="r")
    ax.quiver(*p, *(rot_mat[:, 1] * scale), color="g")
    ax.quiver(*p, *(rot_mat[:, 2] * scale), color="b")
    if name:
        ax.text(*p, name)


def plot_ellipsoid(
    a: np.ndarray,
    t_end: np.ndarray,
    ax: Any,
    color: str = "b",
    label: str | None = None,
    scale: float = 1,
) -> None:
    """Plot a 3D velocity or force ellipsoid at the end-effector position.

    Visualizes the manipulability or dexterity ellipsoid by computing its principal
    axes from eigenvalue decomposition and rotating/translating to tool position.

    Args:
        a: 3x3 ellipsoid matrix (typically Jacobian inverse or similar).
        t_end: End-effector homogeneous transformation matrix (4x4).
        ax: matplotlib 3D axis object.
        color: Color for the ellipsoid wireframe, default "b".
        label: Optional legend label for the ellipsoid.
        scale: Scale factor for the ellipsoid radii, default 1.
    """
    a = np.linalg.inv(a)
    # Position de l'effecteur final
    position = t_end[:3, 3]

    # Décomposition spectrale
    eigvals, eigvecs = np.linalg.eigh(a)

    # anti reflexion : determinant > 0
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1  # Inverser le troisième vecteur propre

    # print("Valeur propre de A (vitesse maxi) : ", eigvals)

    # Rayons = sqrt(valeurs propres)
    radii = np.sqrt(eigvals.real) * scale

    # Paramétrisation sphérique
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotation
    ellipsoid = np.stack((x, y, z), axis=0)
    ellipsoid = np.einsum("ij,jkl->ikl", eigvecs, ellipsoid)

    # Translation vers la position de l'effecteur final
    ellipsoid = ellipsoid + position[:, np.newaxis, np.newaxis]

    # Tracer l'ellipsoïde
    ax.plot_wireframe(
        ellipsoid[0],
        ellipsoid[1],
        ellipsoid[2],
        rstride=4,
        cstride=4,
        color=color,
        alpha=0.7,
        label=label,
    )

    # Ajouter un point pour la légende
    if label:
        ax.scatter(position[0], position[1], position[2], color=color, label=label)


def display_vector(ax: Any, position: np.ndarray, vect: np.ndarray) -> None:
    """Plot a vector as an arrow at a given position.

    Args:
        ax: matplotlib 3D axis object.
        position: 3D position for the arrow origin.
        vect: 3D vector representing the arrow direction and magnitude.
    """
    # Position de l'effecteur final
    ax.quiver(
        position[0],
        position[1],
        position[2],
        vect[0],
        vect[1],
        vect[2],
        color="k",
        linewidth=3,
        arrow_length_ratio=0.1,
    )
