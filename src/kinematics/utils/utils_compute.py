import numpy as np
from numba import njit


@njit(cache=True)
def dh_mat_khalil(
    a: np.ndarray, d: np.ndarray, alpha: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Compute a homogeneous transformation matrix using Denavit-Hartenberg Khalil parameters.

    Constructs a nx4x4 homogeneous transformation matrix from the modified Denavit-Hartenberg
    parameters. Used for building kinematic chains in robotic manipulators.

    Args:
        a: Translation of zi-1 to zi along xi-1 (m).
        d: Translation of xi-1 to xi along zi-1 (m).
        alpha: Rotation of zi-1 to zi around xi-1 (rad).
        theta: Rotation of xi-1 to xi around zi (rad).

    Returns:
        nx4x4 homogeneous transformation matrix.
    """
    n = a.shape[0]
    matrixes = np.zeros((n, 4, 4), dtype=np.float64)

    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    matrixes[:, 0, 0] = cos_theta
    matrixes[:, 1, 0] = cos_alpha * sin_theta
    matrixes[:, 2, 0] = sin_alpha * sin_theta

    matrixes[:, 0, 1] = -sin_theta
    matrixes[:, 1, 1] = cos_alpha * cos_theta
    matrixes[:, 2, 1] = sin_alpha * cos_theta

    matrixes[:, 1, 2] = -sin_alpha
    matrixes[:, 2, 2] = cos_alpha

    matrixes[:, 0, 3] = a
    matrixes[:, 1, 3] = -d * sin_alpha
    matrixes[:, 2, 3] = d * cos_alpha
    matrixes[:, 3, 3] = 1

    return matrixes


@njit(cache=True)
def dh_mat(
    a: np.ndarray, d: np.ndarray, alpha: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Compute a homogeneous transformation matrix using Denavit-Hartenberg parameters.

    Constructs a nx4x4 homogeneous transformation matrix from the modified Denavit-Hartenberg
    parameters. Used for building kinematic chains in robotic manipulators.

    Args:
        a: Translation of zi-1 to zi along xi-1 (m).
        d: Translation of xi-1 to xi along zi-1 (m).
        alpha: Rotation of zi-1 to zi around xi-1 (rad).
        theta: Rotation of xi-1 to xi around zi (rad).

    Returns:
        4x4 homogeneous transformation matrix.
    """
    n = a.shape[0]
    matrixes = np.zeros((n, 4, 4), dtype=np.float64)

    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    mat = np.identity(4)
    matrixes[:, 0, 0] = cos_theta
    matrixes[:, 1, 0] = sin_theta
    matrixes[:, 2, 0] = 0

    matrixes[:, 0, 1] = -sin_theta * cos_alpha
    matrixes[:, 1, 1] = cos_theta * cos_alpha
    matrixes[:, 2, 1] = sin_alpha

    matrixes[:, 0, 2] = sin_alpha * sin_theta
    matrixes[:, 1, 2] = -cos_theta * sin_alpha
    matrixes[:, 2, 2] = cos_alpha

    matrixes[:, 0, 3] = a * cos_theta
    matrixes[:, 1, 3] = a * sin_theta
    matrixes[:, 2, 3] = d
    matrixes[:, 3, 3] = 1

    return matrixes


@njit(cache=True)
def matrix_to_rotvect(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector (axis-angle representation).

    Uses the exponential map to convert from rotation matrix to rotation vector.
    Handles special cases (identity matrix and 180° rotations).

    Args:
        rot_mat: 3x3 rotation matrix.

    Returns:
        Rotation vector of shape (3,) where the direction is the rotation axis
        and magnitude is the rotation angle in radians.
    """
    # Calcul de la trace
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]

    # Sécurité pour acos (bornes explicites)
    if trace < -1.0:
        trace = -1.0
    elif trace > 3.0:
        trace = 3.0

    # Calcul de l'angle theta
    theta: float = np.arccos((trace - 1.0) / 2.0)
    u = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Cas particulier : theta ≈ 0 (matrice identité)
    if abs(theta) < 1e-10:
        return u

    # Cas particulier : theta ≈ pi
    if abs(theta - np.pi) < 1e-10:
        # Trouver l'axe u pour theta = pi
        if (rot_mat[0, 0] + 1) > (rot_mat[1, 1] + 1) and (rot_mat[0, 0] + 1) > (
            rot_mat[2, 2] + 1
        ):
            u = np.array([rot_mat[0, 0] + 1, rot_mat[1, 0], rot_mat[2, 0]])
        elif (rot_mat[1, 1] + 1) > (rot_mat[2, 2] + 1):
            u = np.array([rot_mat[0, 1], rot_mat[1, 1] + 1, rot_mat[2, 1]])
        else:
            u = np.array([rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2] + 1])
        # Normalisation
        norm_u = np.linalg.norm(u)
        u = u / norm_u
        return u * theta

    # Calcul de l'axe u
    u = np.array(
        [
            rot_mat[2, 1] - rot_mat[1, 2],
            rot_mat[0, 2] - rot_mat[2, 0],
            rot_mat[1, 0] - rot_mat[0, 1],
        ]
    )

    # Normalisation de u
    norm_u = np.linalg.norm(u)
    if norm_u > 1e-10:
        u = u / norm_u

    # Retourne le vecteur de rotation (u * theta)
    return u * theta


@njit(cache=True)
def rotvect_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (axis-angle) to a 3x3 rotation matrix.

    Uses Rodrigues' rotation formula to convert from axis-angle representation
    to a rotation matrix. Handles small angles near identity.

    Args:
        rotation_vector: Rotation vector of shape (3,) where direction is axis
                        and magnitude is angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    theta = np.linalg.norm(rotation_vector)
    if theta < 1e-10:  # Si le vecteur est quasi-nul, retourner la matrice identité
        return np.eye(3)

    # Normaliser l'axe de rotation
    u = rotation_vector / theta

    # Composantes de l'axe
    ux, uy, uz = u

    # Calculer les éléments de la matrice de rotation avec la formule de Rodrigues
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos_theta = 1 - cos_theta

    # Matrice de rotation
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = cos_theta + ux * ux * one_minus_cos_theta
    rot_mat[0, 1] = ux * uy * one_minus_cos_theta - uz * sin_theta
    rot_mat[0, 2] = ux * uz * one_minus_cos_theta + uy * sin_theta

    rot_mat[1, 0] = uy * ux * one_minus_cos_theta + uz * sin_theta
    rot_mat[1, 1] = cos_theta + uy * uy * one_minus_cos_theta
    rot_mat[1, 2] = uy * uz * one_minus_cos_theta - ux * sin_theta

    rot_mat[2, 0] = uz * ux * one_minus_cos_theta - uy * sin_theta
    rot_mat[2, 1] = uz * uy * one_minus_cos_theta + ux * sin_theta
    rot_mat[2, 2] = cos_theta + uz * uz * one_minus_cos_theta

    return rot_mat
