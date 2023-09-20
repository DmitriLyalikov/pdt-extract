"""
feature_extract.py

- Author: Dmitri Lyalikov
- Email:  Dlyalikov01@manhattan.edu
- Date of last revision: 05/02/2023
- Status: in development / validating

This module performs characteristic feature extraction on the x and y coordinates of an
edge profile.
These methods derive numerical profile characteristics of the pendant drop:
    - Apex Radius: Found with circle fit approximation
    - Equator Radius
    - Radius_S: Radius at y = 2 * (Equator Radius
    - Drop_Height
    - Capillary Radius
By instantiating a FeatureExtract object with the x, y profile coordinates, all of these features
are automatically saved to a dictionary: self.feature_set as a key, value pair
"""

import numpy as np
from circle_fit import taubinSVD
import matplotlib.pyplot as plt
import pickle
import cv2


class ApexBuilder:
    def __init__(self, x: list[int], y: list[int]):
        """
        :param x: globally used ordered set of x coordinates of the pendant drop profile
        :param y: globally used ordered set of x coordinates of the pendant drop profile
        """

        self.x = x
        self.y = y.sort()


        self.apex_radius = self.find_apex_radius()
        print(self.apex_radius)
        # Normalize to dimensionless ratio to apex radius




    def show_features(self):
        str_features = ""
        for key, value in self.feature_set.items():
            str_features += key + " " + str(value) + " "
        return str_features


    # Use Circle fit to approximate apex radius of edge profile
    # ratio_drop_length: 1 >= float value > 0 representing number points along profile to approximate with
    # change_ro: float value representing minimum value of change in circle radius before stopping approximation
    def find_apex_radius(self, ratio_drop_length: float = 0.2, change_ro: float = .005) -> float:

        num_point_ro_circlefit = round(len(self.x) * ratio_drop_length) + 1

        percent_drop_ro = 0.246
        i = 0
        diff = 0
        r0 = 0
        r_0 = []
        while diff >= change_ro*r0 or num_point_ro_circlefit <= percent_drop_ro * len(self.x):
            points_ro_circlefit = np.stack((self.x[:num_point_ro_circlefit], self.y[:num_point_ro_circlefit]), axis=1)
            xc, yc, r0, sigma = taubinSVD(points_ro_circlefit)
            r_0.append(r0)
            if i > 1:
                diff = abs(r_0[i] - r_0[i-1])
            i += 1
            num_point_ro_circlefit += 1

        return r_0[-1]


    def show_image(self, img):
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()