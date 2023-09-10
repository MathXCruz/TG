from airfoil import Airfoil
import numpy as np


def main():
    airfoils = np.arange(1, 9941, 1)
    airfoils = [
        list(str(airfoil).zfill(4))
        for airfoil in airfoils
        if airfoil % 100 <= 40
    ]
    angles = np.arange(0, 31, 1)
    for airfoil in airfoils:
        for angle in angles:
            Airfoil(airfoil, angle)


if __name__ == '__main__':
    main()
