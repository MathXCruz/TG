import numpy as np
import pandas as pd
from typing import Tuple

class Airfoil():
    def __init__(self, naca: str, theta: int, n_points: int, c: int):
        X, Y, self.t, self.h = self.generate_points(naca, n_points, c)
        X_r, Y_r = self.rotate_airfoil(X, Y, -theta)
        df = pd.DataFrame({'x': X_r, 'y': Y_r})
        df.to_csv(f'./airfoils/{naca[0]}{naca[1]}{naca[2]}{naca[3]}_airfoil_{theta}_degrees.csv', index=False)

    def maximum_wing_thickness(self, c, digit) -> float:
        """ calculate maximum wing thickness
        c: chord length
        digit: naca digits "34" of four series
        """
        return digit * (1 / 100) * c


    # position of the maximum thickness (p) in tenths of chord
    def distance_from_leading_edge_to_max_camber(self, c, digit) -> float:
        """ distance from leading edge to maxiumum wing thickness
        c: chord length
        digit: naca digit "2" of four series
        """
        return digit * (10 / 100) * c


    # maximum camber (m) in percentage of the chord
    def maximum_camber(self, c, digit) -> float:
        """ calculate maximum camber
        c: chord length
        digit: naca digit "1" of four series
        """
        return digit * (1 / 100) * c


    def mean_camber_line_yc(self, m, p, X) -> float:
        """ mean camber line y-coordinates from x = 0 to x = p
        m: maximum camber in percentage of the chord
        p: position of the maximum camber in tenths of chord
        """
        return np.array([
            (m / np.power(p, 2)) * ((2 * p * x) - np.power(x, 2)) if x < p
            else (m / np.power(1 - p, 2)) * (1 - (2 * p) + (2 * p * x) - np.power(x, 2)) for x in X
        ])


    def thickness_distribution(self, t, x) -> np.array:
        """ Calculate the thickness distribution above (+) and below (-) the mean camber line
        t: airfoil thickness
        x: coordinates along the length of the airfoil, from 0 to c
        return: yt thickness distribution
        """
        coeff = t / 0.2
        x1 = 0.2969 * np.sqrt(x)
        x2 = - 0.1260 * x
        x3 = - 0.3516 * np.power(x, 2)
        x4 = + 0.2843 * np.power(x, 3)
        # - 0.1015 coefficient for open trailing edge
        # - 0.1036 coefficient for closed trailing edge
        x5 = - 0.1036 * np.power(x, 4)
        return coeff * (x1 + x2 + x3 + x4 + x5)


    def dyc_dx(self, m, p, X) -> np.array:
        """ derivative of mean camber line with respect to x
        m: maximum camber in percentage of the chord
        p: position of the maximum camber in tenths of chord
        """
        return np.array([
            (2 * m / np.power(p, 2)) * (p - x) if x < p
            else (2 * m / np.power(1 - p, 2)) * (p - x) for x in X
        ])


    def x_upper_coordinates(self, x, yt, theta) -> np.array:
        """ final x coordinates for the airfoil upper surface """
        return x - yt * np.sin(theta)


    def y_upper_coordinates(self, yc, yt, theta) -> np.array:
        """ final y coordinates for the airfoil upper surface """
        return yc + yt * np.cos(theta)


    def x_lower_coordinates(self, x, yt, theta) -> np.array:
        """ final x coordinates for the airfoil lower surface """
        return x + yt * np.sin(theta)


    def y_lower_coordinates(self, yc, yt, theta) -> np.array:
        """ final y coordinates for the airfoil lower surface """
        return yc - yt * np.cos(theta)


    def generate_points(self, naca: str, n_points: str, c: int) -> Tuple[np.array, np.array, float, float]:
        digits = [int(d) for d in naca]
        X = np.linspace(0, 1, int(n_points/2))
        max_camber_distance = self.distance_from_leading_edge_to_max_camber(c, digits[1])
        max_camber = self.maximum_camber(c, digits[0])
        max_wing_thickness = self.maximum_wing_thickness(c, digits[2]*10+digits[3])
        y_c = self.mean_camber_line_yc(max_camber, max_camber_distance, X)
        y_t = self.thickness_distribution(max_wing_thickness, X)
        gradient = self.dyc_dx(max_camber, max_camber_distance, X)
        theta = np.arctan(gradient)
        x_u = self.x_upper_coordinates(X, y_t, theta)
        x_l = self.x_lower_coordinates(X, y_t, theta)
        y_u = self.y_upper_coordinates(y_c, y_t, theta)
        y_l = self.y_lower_coordinates(y_c, y_t, theta)
        return np.append(np.flip(x_u), x_l), np.append(np.flip(y_u), y_l), max_wing_thickness, max_camber


    def rotate_airfoil(self, X: np.array, Y: np.array, theta: float) -> Tuple[np.array, np.array]:
        theta = theta/180*np.pi
        xr = [xi*np.cos(theta) - yi*np.sin(theta) for xi, yi in zip(X,Y)]
        yr = [xi*np.sin(theta) + yi*np.cos(theta) for xi, yi in zip(X,Y)]
        return xr, yr
