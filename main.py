from airfoil import Airfoil
import numpy as np
from panel_method import (
    define_panels,
    Freestream,
    calculate_variables,
    calculate_gamma,
    assign_pressure_coefficient,
    assign_tangential_velocity,
    calculate_lift_coefficient,
    calculate_moment_coefficient,
)
from itertools import chain
from finite_element_method import (
    calculate_displacements,
    format_stiffness_matrix,
)
import time


def main():

    try:
        c = float(input('Chord length [m] = '))
    except ValueError:
        c = 1
    airfoil = input('NACA airfoil series = ') or '4411'
    try:
        E = float(input('Elasticity Modulus [GPa] = ')) * 10e9
    except ValueError:
        E = 70 * 10e9
    try:
        G = float(input('Shear Modulus [GPa] = ')) * 10e9
    except ValueError:
        G = 26.5 * 10e9
    try:
        span = float(input('Wings span [m] = '))
    except ValueError:
        span = 5
    try:
        N = int(input('Amount of wing structural nodes = '))
    except ValueError:
        N = 100
    try:
        alpha = float(input('Initial angle of attack [°] = '))
    except ValueError:
        alpha = 5
    try:
        Npanels = int(input('Amount of panels in airfoil = '))
    except ValueError:
        Npanels = 20
    try:
        rho = float(input('Air density [Kg/m³] = '))
    except ValueError:
        rho = 1.225
    try:
        u_inf = float(input('Airflow speed [m/s] = '))
    except ValueError:
        u_inf = 155

    print(
        f'\nInput parameters:\nc = {c}, NACA = {airfoil}, E = {E}, G = {G}, span = {span}, N = {N}, alpha0 = {alpha}, '
        f'Npanels = {Npanels}, rho = {rho}, u_inf = {u_inf}\n'
    )

    d_torsion = [1]
    i = 0
    alpha = alpha * np.ones(N)
    max_torsion = alpha[-1] - alpha[0]

    a = Airfoil(airfoil, 0, Npanels * 100, c)
    x = np.array(a.X_r)
    y = np.array(a.Y_r)
    I = 0.036 * c * a.t * (a.t**2 + a.h**2)
    d = a.t / 2
    J = np.pi * d**4 / 32
    e = 0.1 * c
    section_length = span / (N - 1)
    S = section_length * c
    panels = [define_panels(x, y, Npanels) for _ in range(N)]
    K = format_stiffness_matrix(E, G, N, I, J, section_length)

    while max_torsion < 90 and abs(d_torsion[-1]) > 0.5:
        start_time = time.time()

        freestream = [Freestream(u_inf, a) for a in alpha]
        variables = calculate_variables(freestream, panels)

        gamma = calculate_gamma(panels, variables)

        assign_tangential_velocity(freestream, panels, gamma)

        assign_pressure_coefficient(freestream, panels)

        airfoils = [
            Airfoil(airfoil, f.alpha, Npanels * 100, c) for f in freestream
        ]
        X = [np.array(a.X_r) for a in airfoils]
        Y = [np.array(a.Y_r) for a in airfoils]
        panels = [define_panels(x, y, Npanels) for (x, y) in zip(X, Y)]
        Cl = calculate_lift_coefficient(freestream, panels, gamma, airfoils)
        Cm = calculate_moment_coefficient(panels)

        L = [
            cl * S * rho * (np.cos(f.alpha) * f.u_inf) ** 2 * 0.5
            for (cl, f) in zip(Cl, freestream)
        ]
        Mf = [l * s for (l, s) in zip(L, range(N))]
        M = [
            cm * S * rho * (np.cos(f.alpha) * f.u_inf) ** 2 * 0.5
            for (cm, f) in zip(Cm, freestream)
        ]
        T = [m + e * l for (l, m) in zip(L, M)]
        F = np.array(list((chain.from_iterable(zip(L, Mf, T)))))

        d_torsion = calculate_displacements(N, F, K)
        alpha = alpha + d_torsion
        max_torsion = abs(alpha[-1] - alpha[0])
        i += 1
        print(
            f'Iteration {i} finished in {time.time() - start_time} seconds with alpha = {alpha}, max_torsion = '
            f'{max_torsion} and torsion on last node = {d_torsion[-1]}'
        )
        if max_torsion >= 90:
            print('\nDivergence occurred!!')
        if abs(d_torsion[-1]) <= 0.5:
            print(f'\nThe simulation converged, max torsion = {max_torsion}°.')


if __name__ == '__main__':
    main()
