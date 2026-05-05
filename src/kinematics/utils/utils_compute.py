import numpy as np
from numba import njit


# Kahlil
@njit(cache=True)
def dh_mat(d, r, alpha, theta):
    mat = np.identity(4)
    mat[0, 0] = np.cos(theta)
    mat[1, 0] = np.cos(alpha)*np.sin(theta)
    mat[2, 0] = np.sin(alpha)*np.sin(theta)

    mat[0, 1] = -np.sin(theta)
    mat[1, 1] = np.cos(alpha)*np.cos(theta)
    mat[2, 1] = np.sin(alpha)*np.cos(theta)
    
    mat[1, 2] = -np.sin(alpha)
    mat[2, 2] = np.cos(alpha)

    mat[0, 3] = d
    mat[1, 3] = -r*np.sin(alpha)
    mat[2, 3] = r*np.cos(alpha)

    return mat

@njit(cache=True)
def matrix_to_rotvect(R):
    # Calcul de la trace
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    # Sécurité pour acos (bornes explicites)
    if trace < -1.0:
        trace = -1.0
    elif trace > 3.0:
        trace = 3.0

    # Calcul de l'angle theta
    theta = np.arccos((trace - 1.0) / 2.0)

    # Cas particulier : theta ≈ 0 (matrice identité)
    if abs(theta) < 1e-10:
        return np.array([0.0, 0.0, 0.0])

    # Cas particulier : theta ≈ pi
    if abs(theta - np.pi) < 1e-10:
        # Trouver l'axe u pour theta = pi
        if (R[0, 0] + 1) > (R[1, 1] + 1) and (R[0, 0] + 1) > (R[2, 2] + 1):
            u = np.array([R[0, 0] + 1, R[1, 0], R[2, 0]])
        elif (R[1, 1] + 1) > (R[2, 2] + 1):
            u = np.array([R[0, 1], R[1, 1] + 1, R[2, 1]])
        else:
            u = np.array([R[0, 2], R[1, 2], R[2, 2] + 1])
        # Normalisation
        norm_u = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
        u = u / norm_u
        return u * theta

    # Calcul de l'axe u
    u = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    # Normalisation de u
    norm_u = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    if norm_u > 1e-10:
        u = u / norm_u

    # Retourne le vecteur de rotation (u * theta)
    return u * theta


@njit(cache=True)
def rotvect_to_matrix(rotation_vector):
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
    R = np.zeros((3, 3))
    R[0, 0] = cos_theta + ux * ux * one_minus_cos_theta
    R[0, 1] = ux * uy * one_minus_cos_theta - uz * sin_theta
    R[0, 2] = ux * uz * one_minus_cos_theta + uy * sin_theta

    R[1, 0] = uy * ux * one_minus_cos_theta + uz * sin_theta
    R[1, 1] = cos_theta + uy * uy * one_minus_cos_theta
    R[1, 2] = uy * uz * one_minus_cos_theta - ux * sin_theta

    R[2, 0] = uz * ux * one_minus_cos_theta - uy * sin_theta
    R[2, 1] = uz * uy * one_minus_cos_theta + ux * sin_theta
    R[2, 2] = cos_theta + uz * uz * one_minus_cos_theta

    return R