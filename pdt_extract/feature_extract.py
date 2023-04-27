"""
feature_extract.py
Author: Dmitri Lyalikov
Email: Dlyalikov01@Manhattan.edu
Date: 4/27/2023
These methods derive numerical profile characteristics of the pendant drop
    - Apex Radius: Found with circle fit approximation
    - Equator Radius
    - Radius_S: Radius at y = 2 * (Equator Radius
"""

import numpy as np
from circle_fit import taubinSVD


class FeatureExtract:
    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y
        self.equator_radius = 0
        self.s_radius = 0
        self.capillary_radius = self.x[-1]
        self.drop_height = self.y[-1]
        self.equator_radius, self.s_radius = self.find_re_rs(5)
        print(f"Equator radius: {self.equator_radius},"
              f"S radius: {self.s_radius},"
              f"Capillary radius: {self.capillary_radius},"
              f"Drop Height: {self.drop_height}")

    def average_x(self, i: int, n: int) -> int:
        s = 0
        for j in range(i-n, i+n+1):
            s = s + self.x[j]
            return s / (2 * n + 1)

    def recursive_equator_radius(self, i, n):
        # use recursive approach: start from apex, continue until x decreases
        # at i-th point we average x of x-n to x+n to suppress noise
        # compare x-i_th vs x_i+t_th until it decreases to find equator
        if self.average_x(i, n) < self.average_x(i+1, n) and i <= (len(self.x) - n-3):
            i += 1
            output = self.recursive_equator_radius(i, n)
            if output is not None:
                self.equator_radius = output
            else:
                if i <= len(self.x) * .7:
                    # assumed 70% of drop is enough for the equator radius
                    return self.x[i]
                else:
                    return

    def find_re_rs(self, drop_height: int, n=5) -> (int, int):
        # Finding Equator Radius (Re) and Rs @ y=2Re
        """
        :param n:
        :return: tuple (equator_radius: int, s_radius: int)
        """
        i = n
        # A recursive function that returns equator radius
        self.recursive_equator_radius(i, n)
        if self.equator_radius is 0:
            # equator radius is 0: drop is not well-deformed: Beta>0.7
            # find equator radius from circle fitting
            # select 40% of the total number of points for circle fitting
            num_points_to_circlefit = round(0.4 * len(self.x))
            points_rh_circlefit = np.stack(
                (self.x[:num_points_to_circlefit],
                 self.y[:num_points_to_circlefit]), axis=1)
            xc, yc, self.equator_radius, sigma = taubinSVD(points_rh_circlefit)

        # Find s_radius at y = 2 * equator_radius
        if self.equator_radius < 0.5 * drop_height:
            # res = index of y if y > 2 * equator_radius
            res = next(xx for xx, val in enumerate(self.y) if val > 2 * self.equator_radius)
            self.s_radius = self.x[res]
        else:
            # Drop is too small
            self.s_radius = self.capillary_radius
        return self.equator_radius, self.s_radius


# divide all elements R0
