"""
extraction_methods.py

- Author: Dmitri Lyalikov
- Email:  Dlyalikov01@manhattan.edu
- Date of last revision: 07/16/2023
- Status: init

This script is used to cross-check the functional behavior of the pendant drop parameter extraction functions developed
in pdt_extract/feature_extract
"""

from circle_fit import taubinSVD
import numpy as np
from scipy.integrate import odeint
import math
import pandas as pd


# Original method to find R-S, R-E, Drop Height
# Finding Equator radius (Re) and Rs @ y=2Re
def Find_Re_Rs(x, y, n, Drop_Height):
    # at i_th x: we average from i-n to i+n which means 11 points
    global R_e
    R_e = 0
    i = n

    def Average_x(x, i, n):
        # i-n>=0
        s = 0
        for j in range(i - n, i + n + 1):
            s = s + x[j]
        return s / (2 * n + 1)

    def Recur_Equator_Radius(x, i, n):
        global R_e
        # R_e is the equator radius: must be defined here not outside
        # We use recursive approach: Start from Apex continue until x decreases
        # At i-th point we averagne x of x-n to x+n to subpress noise
        # We compare x_i_th vs x_i+1_th until it decrease to find equator
        if Average_x(x, i, n) < Average_x(x, i + 1, n) and i <= len(x) - n - 3:
            i = i + 1
            Output = Recur_Equator_Radius(x, i, n)
            if Output is not None:
                # Since recursive returns None!!! we use global Variable
                R_e = Output
        else:
            if i <= len(x) * 0.7:
                # I assumed 70% of drop is enough for R_e
                # print(i)
                # print(x[i])
                return x[i]
            else:
                return

    # A recursive function that returns equator radius
    Recur_Equator_Radius(x, i, n)

    if R_e == 0:
        # R_e=0: drop is not well-deformed e.g. Beta>0.7. Find R_e from cirle fitting
        # I selected 40% of the total number of points for circle fitting
        num_point_RH_Circlefit = round(0.4 * len(x))
        Points_RH_Circlefit = np.stack((x[:num_point_RH_Circlefit], y[:num_point_RH_Circlefit]), axis=1)
        xc, yc, R_e, sigma = taubinSVD(Points_RH_Circlefit)
    # Find R_s at y=2*R_e
    if R_e < 0.5 * Drop_Height:
        # res=index of y if y>2*R_e
        res = next(xx for xx, val in enumerate(y) if val > 2 * R_e)
        R_s = x[res]
    else:
        # Drop is too small
        R_s = R_Cap
    return R_e, R_s

x, y = Add_Noise_Drop_Profile(z, noise_Percent_of_datamean)
# Generating outputs
Drop_Height = y[-1]  # Drop Height
R_Cap = x[-1]  # Capillary radius