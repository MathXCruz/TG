import numpy as np


def calculate_displacements(E, G, N, I, J, section_length, F):

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
