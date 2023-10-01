import numpy as np


def calculate_displacements(N: int, F: np.array, K: np.array) -> np.array:
    """Calculate the torsional displacement on the nodes.

    Args:
        N (int): The amount of nodes along the beam
        F (np.array): The forces applied on the beam
        K (np.array): The global stiffness matrix

    Returns:
        np.array: The torsional displacements in degrees
    """
    d_known = np.array([0, 0, 0])
    known_indices = np.array([0, 1, 2])
    unknown_indices = np.setdiff1d(np.arange(len(F)), known_indices)
    Kuk = K[np.ix_(unknown_indices, known_indices)]
    Kuu = K[np.ix_(unknown_indices, unknown_indices)]
    Fu = F[unknown_indices]
    d_unknown = np.linalg.solve(Kuu, Fu - np.dot(Kuk, d_known))
    d_full = np.zeros(len(F))
    d_full[known_indices] = d_known
    d_full[unknown_indices] = d_unknown
    d_torsion = d_full[2 : N * 3 : 3] * 180 / np.pi

    return d_torsion


def format_stiffness_matrix(
    E: float, G: float, N: int, I: float, J: float, section_length: float
) -> np.array:
    """_summary_

    Args:
        E (float): Elasticity Modulus
        G (float): Shear Modulus
        N (int): The amount of nodes along the beam
        I (float): Moment of Intertia
        J (float): Polar Moment of Intertia
        section_length (float): The size of each element between two nodes

    Returns:
        np.array: The formatted global stiffness matrix
    """
    k = np.asarray(
        [
            [
                12 * E * I / section_length**3,
                6 * E * I / section_length**2,
                0,
                12 * E * I / section_length**3,
                6 * E * I / section_length**2,
                0,
            ],
            [
                6 * E * I / section_length**2,
                4 * E * I / section_length,
                0,
                -6 * E * I / section_length**2,
                2 * E * I / section_length,
                0,
            ],
            [0, 0, G * J / section_length, 0, 0, -G * J / section_length],
            [
                -12 * E * I / section_length**3,
                -6 * E * I / section_length**2,
                0,
                12 * E * I / section_length**3,
                -6 * E * I / section_length**2,
                0,
            ],
            [
                6 * E * I / section_length**2,
                2 * E * I / section_length,
                0,
                -6 * E * I / section_length**2,
                4 * E * I / section_length,
                0,
            ],
            [0, 0, -G * J / section_length, 0, 0, G * J / section_length],
        ],
        dtype=np.float32,
    )

    K = np.zeros(((N) * 3, (N) * 3))

    for m in range(N - 1):
        for i in range(len(k[0])):
            for j in range(len(k[1])):
                K[i + 3 * m, j + 3 * m] += k[i, j]
    return K
