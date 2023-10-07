from scipy import integrate
import numpy as np
import math

from airfoil import Airfoil


class Panel:
    """
    Contains information related to a panel.
    """

    def __init__(self, xa: float, ya: float, xb: float, yb: float):
        """
        Initializes the panel.

        Sets the end-points and calculates the center, length,
        and angle (with the x-axis) of the panel.
        Defines if the panel is on the lower or upper surface of the geometry.
        Initializes the source-sheet strength, tangential velocity,
        and pressure coefficient to zero.

        Parameters
        ----------
        xa: float
            x-coordinate of the first end-point.
        ya: float
            y-coordinate of the first end-point.
        xb: float
            x-coordinate of the second end-point.
        yb: float
            y-coordinate of the second end-point.
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa + xb) / 2, (
            ya + yb
        ) / 2  # control-point (center-point)
        self.length = math.sqrt(
            (xb - xa) ** 2 + (yb - ya) ** 2
        )  # length of the panel

        # orientation of the panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = math.acos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = math.pi + math.acos(-(yb - ya) / self.length)

        # location of the panel
        if self.beta <= math.pi:
            self.loc = 'extrados'
        else:
            self.loc = 'intrados'

        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient

    def __repr__(self) -> str:
        return f'xa={self.xa}, ya={self.ya}, xc={self.xc}, yc={self.yc}, length={self.length}, sigma={self.sigma}, vt={self.vt}, cp={self.cp}, loc={self.loc}'


def define_panels(x: np.array, y: np.array, N: int) -> np.array:
    """
    Discretizes the geometry into panels using the 'cosine' method.

    Parameters
    ----------
    x: 1D array of floats
        x-coordinate of the points defining the geometry.
    y: 1D array of floats
        y-coordinate of the points defining the geometry.
    N: integer, optional
        Number of panels;

    Returns
    -------
    panels: 1D np array of Panel objects
        The discretization of the geometry into panels.
    """
    R = (max(x) - min(x)) / 2  # radius of the circle
    x_center = (max(x) + min(x)) / 2  # x-coord of the center
    # define x-coord of the circle points
    x_circle = x_center + R * np.cos(np.linspace(0.0, 2 * math.pi, N + 1))

    x_ends = np.copy(x_circle)  # projection of the x-coord on the surface
    y_ends = np.empty_like(x_ends)  # initialization of the y-coord np array

    x, y = np.append(x, x[0]), np.append(
        y, y[0]
    )  # extend arrays using np.append

    # computes the y-coordinate of end-points
    I = 0
    for i in range(N):
        while I < len(x) - 2:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (
                x[I + 1] <= x_ends[i] <= x[I]
            ):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]

    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels


class Freestream:
    """Freestream conditions."""

    def __init__(self, u_inf: float = 1.0, alpha: float = 0.0):
        """Sets the freestream conditions.

        Arguments
        ---------
        u_inf -- Farfield speed (default 1.0).
        alpha -- Angle of attack in degrees (default 0.0).
        """
        self.u_inf = u_inf
        self.alpha = alpha * math.pi / 180          # degrees --> radians


def integral(
    x: float, y: float, panel: Panel, dxdz: float, dydz: float
) -> float:
    """
    Evaluates the contribution of a panel at one point.

    Parameters
    ----------
    x: float
        x-coordinate of the target point.
    y: float
        y-coordinate of the target point.
    panel: Panel object
        Source panel which contribution is evaluated.
    dxdz: float
        Derivative of x in the z-direction.
    dydz: float
        Derivative of y in the z-direction.

    Returns
    -------
    Integral over the panel of the influence at the given target point.
    """

    def integrand(s: float) -> float:
        return (
            (x - (panel.xa - math.sin(panel.beta) * s)) * dxdz
            + (y - (panel.ya + math.cos(panel.beta) * s)) * dydz
        ) / (
            (x - (panel.xa - math.sin(panel.beta) * s)) ** 2
            + (y - (panel.ya + math.cos(panel.beta) * s)) ** 2
        )

    return integrate.quad(integrand, 0.0, panel.length)[0]


def source_matrix(panels: list[Panel]) -> np.array:
    """Builds the source matrix.

    Arguments
    ---------
    panels -- array of panels.

    Returns
    -------
    A -- NxN matrix (N is the number of panels).
    """
    N = len(panels)
    A = np.empty((N, N), dtype=float)
    np.fill_diagonal(A, 0.5)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i, j] = (
                    0.5
                    / math.pi
                    * integral(
                        p_i.xc,
                        p_i.yc,
                        p_j,
                        math.cos(p_i.beta),
                        math.sin(p_i.beta),
                    )
                )

    return A


def vortex_array(panels: list[Panel]) -> np.array:
    """Builds the vortex array.

    Arguments
    ---------
    panels - array of panels.

    Returns
    -------
    a -- 1D array (Nx1, N is the number of panels).
    """
    a = np.zeros(len(panels), dtype=float)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                a[i] -= (
                    0.5
                    / math.pi
                    * integral(
                        p_i.xc,
                        p_i.yc,
                        p_j,
                        +math.sin(p_i.beta),
                        -math.cos(p_i.beta),
                    )
                )

    return a


def kutta_array(panels: list[Panel]) -> np.array:
    """Builds the Kutta-condition array.

    Arguments
    ---------
    panels -- array of panels.

    Returns
    -------
    a -- 1D array (Nx1, N is the number of panels).
    """
    N = len(panels)
    a = np.zeros(N + 1, dtype=float)

    a[0] = (
        0.5
        / math.pi
        * integral(
            panels[N - 1].xc,
            panels[N - 1].yc,
            panels[0],
            -math.sin(panels[N - 1].beta),
            +math.cos(panels[N - 1].beta),
        )
    )
    a[N - 1] = (
        0.5
        / math.pi
        * integral(
            panels[0].xc,
            panels[0].yc,
            panels[N - 1],
            -math.sin(panels[0].beta),
            +math.cos(panels[0].beta),
        )
    )

    for i, panel in enumerate(panels[1 : N - 1]):
        a[i] = (
            0.5
            / math.pi
            * (
                integral(
                    panels[0].xc,
                    panels[0].yc,
                    panel,
                    -math.sin(panels[0].beta),
                    +math.cos(panels[0].beta),
                )
                + integral(
                    panels[N - 1].xc,
                    panels[N - 1].yc,
                    panel,
                    -math.sin(panels[N - 1].beta),
                    +math.cos(panels[N - 1].beta),
                )
            )
        )

        a[N] -= (
            0.5
            / math.pi
            * (
                integral(
                    panels[0].xc,
                    panels[0].yc,
                    panel,
                    +math.cos(panels[0].beta),
                    +math.sin(panels[0].beta),
                )
                + integral(
                    panels[N - 1].xc,
                    panels[N - 1].yc,
                    panel,
                    +math.cos(panels[N - 1].beta),
                    +math.sin(panels[N - 1].beta),
                )
            )
        )

    return a


def build_matrix(panels: list[Panel]) -> np.array:
    """Builds the matrix of the linear system.

    Arguments
    ---------
    panels -- array of panels.

    Returns
    -------
    A -- (N+1)x(N+1) matrix (N is the number of panels).
    """
    N = len(panels)
    A = np.empty((N + 1, N + 1), dtype=float)

    AS = source_matrix(panels)
    av = vortex_array(panels)
    ak = kutta_array(panels)

    A[0:N, 0:N], A[0:N, N], A[N, :] = AS[:, :], av[:], ak[:]

    return A


def build_rhs(panels: list[Panel], freestream: Freestream) -> np.array:
    """Builds the RHS of the linear system.

    Arguments
    ---------
    panels -- array of panels.
    freestream -- farfield conditions.

    Returns
    -------
    b -- 1D array ((N+1)x1, N is the number of panels).
    """
    N = len(panels)
    b = np.empty(N + 1, dtype=float)

    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * math.cos(freestream.alpha - panel.beta)
    b[N] = -freestream.u_inf * (
        math.sin(freestream.alpha - panels[0].beta)
        + math.sin(freestream.alpha - panels[N - 1].beta)
    )

    return b


def get_tangential_velocity(
    panels: list[Panel], freestream: Freestream, gamma: float
):
    """Computes the tangential velocity on the surface.

    Arguments
    ---------
    panels -- array of panels.
    freestream -- farfield conditions.
    gamma -- circulation density.
    """
    N = len(panels)
    A = np.zeros((N, N + 1))

    for i, p_i in enumerate(panels):
        x_minus_xa = freestream.u_inf - p_i.xa
        y_minus_ya = freestream.u_inf - p_i.ya

        integrand1 = (
            x_minus_xa - np.sin(p_i.beta) * np.linspace(0, p_i.length, N)
        ) * (-np.sin(p_i.beta)) + (
            y_minus_ya + np.cos(p_i.beta) * np.linspace(0, p_i.length, N)
        ) * np.cos(
            p_i.beta
        )

        integrand2 = (
            x_minus_xa - np.sin(p_i.beta) * np.linspace(0, p_i.length, N)
        ) * np.cos(p_i.beta) + (
            y_minus_ya + np.cos(p_i.beta) * np.linspace(0, p_i.length, N)
        ) * np.sin(
            p_i.beta
        )

        A[i, :N] = (
            0.5 / math.pi * integrate.simps(integrand1, dx=p_i.length / N)
        )
        A[i, N] = (
            -0.5 / math.pi * integrate.simps(integrand2, dx=p_i.length / N)
        )

    b = freestream.u_inf * np.sin(
        [freestream.alpha - panel.beta for panel in panels]
    )

    var = np.append([panel.sigma for panel in panels], gamma)

    vt = np.dot(A, var) + b
    for i, panel in enumerate(panels):
        panel.vt = vt[i]


def get_pressure_coefficient(panels: list[Panel], freestream: Freestream):
    """Computes the surface pressure coefficients.

    Arguments
    ---------
    panels -- array of panels.
    freestream -- farfield conditions.
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf) ** 2


def calculate_variables(
    freestream: list[Freestream], panels: list[list[Panel]]
) -> np.array:
    A = [build_matrix(p) for p in panels]
    B = [build_rhs(p, f) for f in freestream for p in panels]
    variables = [np.linalg.solve(a, b) for (a, b) in zip(A, B)]
    return variables


def assign_pressure_coefficient(
    freestream: list[Freestream], panels: list[list[Panel]]
):
    for (p, f) in zip(panels, freestream):
        get_pressure_coefficient(p, f)


def assign_tangential_velocity(
    freestream: list[Freestream], panels: list[list[Panel]], gamma: list[float]
):
    for (p, f, g) in zip(panels, freestream, gamma):
        get_tangential_velocity(p, f, g)


def calculate_gamma(
    panels: list[list[Panel]], variables: np.array
) -> np.array:
    gamma = np.array(0)
    for p in panels:
        for i, panel in enumerate(p):
            panel.sigma = variables[0][i]
        gamma = [v[-1] for v in variables]
    return gamma


def calculate_moment_coefficient(panels: list[list[Panel]]) -> list[float]:
    Cm = []
    for p in panels:
        Cm.append(
            sum(
                panel.cp
                * (panel.xc - 0.25)
                * panel.length
                * np.cos(panel.beta)
                for panel in p
            )
        )
    return Cm


def calculate_lift_coefficient(
    freestream: list[Freestream],
    panels: list[list[Panel]],
    gamma: list[float],
    airfoils: list[Airfoil],
) -> list[float]:
    Cl = []
    for (f, g, p, a) in zip(freestream, gamma, panels, airfoils):
        Cl.append(
            g
            * sum(panel.length for panel in p)
            / (0.5 * f.u_inf * (max(a.X_r) - min(a.X_r)))
        )
    return Cl
