import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robot_arm_kinematics.utils import matrix_to_rotvect, rotvect_to_matrix


def is_valid_rotation_matrix(R: np.ndarray) -> bool:
    # Vérifier que R est carrée 3x3
    if R.shape != (3, 3):
        return False

    # Vérifier l'orthogonalité (R^T R = I)
    RT_R = np.dot(R.T, R)
    I = np.eye(3)
    if not np.allclose(RT_R, I, atol=1e-6):
        return False

    # Vérifier que le déterminant est 1
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        return False

    return True


def main() -> None:
    N = 1000  # number of tests
    # test rotvector to matrix
    for i in tqdm(range(N)):
        rotation = R.from_euler(
            "xyz",
            [
                np.random.randint(0, 360),
                np.random.randint(0, 360),
                np.random.randint(0, 360),
            ],
            degrees=True,
        )
        matrix = rotation.as_matrix()

        if not is_valid_rotation_matrix(matrix):
            print(R)
            pass

        rotvec = matrix_to_rotvect(matrix)
        scipy_rotvec = rotation.as_rotvec()

        if np.allclose(rotvec, scipy_rotvec, rtol=1e-6, atol=1e-6) == False:
            print(matrix)
            print(rotvec)
            print(scipy_rotvec)
            exit(-1)

    # test matrix to rotvect
    for i in tqdm(range(N)):
        rotation = R.from_euler(
            "xyz",
            [
                np.random.randint(0, 360),
                np.random.randint(0, 360),
                np.random.randint(0, 360),
            ],
            degrees=True,
        )

        rotvec = rotation.as_rotvec()

        matrix = rotvect_to_matrix(rotvec)
        scipy_matrix = rotation.as_matrix()

        if np.allclose(matrix, scipy_matrix, rtol=1e-6, atol=1e-6) == False:
            print(rotvec)
            print(matrix)
            print(scipy_matrix)
            exit(-1)


if __name__ == "__main__":
    main()
