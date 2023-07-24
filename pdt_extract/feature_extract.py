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


class FeatureExtract:
    def __init__(self, x: list[int], y: list[int]):
        """
        :param x: globally used ordered set of x coordinates of the pendant drop profile
        :param y: globally used ordered set of x coordinates of the pendant drop profile
        """

        # dbg_log = file.open("FeatureExtract-Dbg-log.txt", 'a')


        self.x = x
        self.y = y
        print(len(x))
        #self.capillary_radius = self.x[-1]
        self.capillary_radius = self.x[-1]
        #self.drop_height = self.y[-1]
        self.drop_height = self.y[0]
        self.equator_radius, self.s_radius = self.find_re_rs()
        # self.s_radius = self.find_s_radius(equator_index)
        self.apex_radius = self.find_apex_radius()
        #self.print_debug_log("After init")
        # Normalize to dimensionless ratio to apex radius
        self.feature_set = {
            "Drop height": self.drop_height / self.apex_radius,
            "Capillary radius": self.capillary_radius / self.apex_radius,
            "R-s": self.s_radius / self.apex_radius,
            "R-e": self.equator_radius / self.apex_radius,
            "Apex Radius": self.apex_radius
        }
        print(f"Apex radius (Pixels): {self.apex_radius }")
        print(f"Equator radius: {self.equator_radius }\n"
              f"S radius: {self.s_radius }\n"
              f"Capillary radius: {self.capillary_radius}\n"
              f"Drop Height: {self.drop_height }")

    def show_features(self):
        str_features = ""
        for key, value in self.feature_set.items():
            str_features += key + " " + str(value) + " "
        return str_features

    def average_x(self, i: int, n: int) -> int:
        s = 0
        for j in range(i-n, i+n+1):
            s = s + self.x[j]
        return s / (2 * n + 1)

    def recursive_equator_radius(self, i, n):
        # use recursive approach: start from apex, continue until x decreases
        # at i-th point we average x of x-n to x+n to suppress noise
        # compare x-i_th vs x_i+t_th until it decreases to find equator
        if self.average_x(i, n) < self.average_x(i+1, n) and i <= len(self.x) - n-3:
            i += 1
            output = self.recursive_equator_radius(i, n)
            if output is not None:
                self.equator_radius = output
        else:
            if i <= len(self.x) * 0.7:
                # assumed 70% of drop is enough for the equator radius
                return self.x[i]
            else:
                return

    def find_re_rs(self, n=5) -> (int, int):
        # Finding Equator Radius (Re) and Rs @ y=2Re
        """
        ;param self:
        :param n:
        :return: tuple (equator_radius: int, s_radius: int)
        """

        i = n
        self.equator_radius = 0
        # A recursive function that returns equator radius
        self.recursive_equator_radius(i, n)
        if self.equator_radius == 0:

            # equator radius is 0: drop is not well-deformed: Beta>0.7
            # find equator radius from circle fitting
            # select 40% of the total number of points for circle fitting
            num_points_to_circlefit = round(0.4 * len(self.x))
            points_rh_circlefit = np.stack(
                (self.x[:num_points_to_circlefit],
                 self.y[:num_points_to_circlefit]), axis=1)
            xc, yc, self.equator_radius, sigma = taubinSVD(points_rh_circlefit)

        # Find s_radius at y = 2 * equator_radius
        if self.equator_radius < 0.5 * self.drop_height:
            # res = index of y if y > 2 * equator_radiuso
            res = next(xx for xx, val in enumerate(self.y) if val > 2 * self.equator_radius)
            self.s_radius = self.x[res]
        else:
            # Drop is too small
            self.s_radius = self.capillary_radius
        return self.equator_radius, self.s_radius

    # Use Circle fit to approximate apex radius of edge profile
    # ratio_drop_length: 1 >= float value > 0 representing number points along profile to approximate with
    # change_ro: float value representing minimum value of change in circle radius before stopping approximation
    def find_apex_radius(self, ratio_drop_length: float = 0.15, change_ro: float = .005) -> float:

        num_point_ro_circlefit = round(len(self.x) * ratio_drop_length) + 1

        percent_drop_ro = 0.1
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

    # Find maximum (bulge) x value from 70% of profile
    def find_equator_radius(self):
        split = int(len(self.x) * 0.7)

        # Slice the first 70% of the list from bottom and find the maximum value
        return max(self.x[:split]), np.argmax(self.x[:split])

    # Find radius at point X = [2 * equator_radius] between capillary and equator
    def find_s_radius(self, equator_index):
        print(equator_index)
        rs_index = ((len(self.x) - equator_index) / 2) + equator_index
        print(rs_index)
        return self.x[int(rs_index)]

    # def print_debug_log(self, step: str, log):


"""
gen_training_data.py
"""


from circle_fit import taubinSVD
import numpy as np
from scipy.integrate import odeint
import math
import pandas as pd
from feature_extract import FeatureExtract

# All functions are defined in this block

def Drop_Profil(z, t, Beta):
    x = z[0]
    y = z[1]
    phi = z[2]
    dxdt = math.cos(phi)
    dydt = math.sin(phi)
    dphidt = 2 - Beta * y - math.sin(phi) / x
    dzdt = [dxdt, dydt, dphidt]
    return dzdt


# Adding noise to the drop profile
def Add_Noise_Drop_Profile(z, noise_Percent_of_datamean):
    y = z[:, 1]
    x = z[:, 0]
    noise = np.random.normal(0, 1, len(x))  # normal distribution mean=0 STD=1
    # Adding fraction of data average as noise
    x = x + noise / noise.max() * noise_Percent_of_datamean * x.mean()
    y = y + noise / noise.max() * noise_Percent_of_datamean * y.mean()
    # Sorting data from ymin to ymax for further analysis
    loc_y_incr = np.argsort(y)
    x = x[loc_y_incr]  # sorted from apex
    y = y[loc_y_incr]  # sorted from apex
    return x, y


# Finding Equator radius (Re) and Rs @ y=2Re
def Find_Re_Rs(x, y, n, Drop_Height):
    """
    print(x)
    print(y)
    print(n)
    print(Drop_Height)
    """
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


# This function returns Smin and Smax for integration based on Beta values
# Developed based on Helen, Payton and Dmitri Excel file
def find_Smin_Smax(Beta, SF):
    # SF is a safety factor. we report (1+SF)*Smin and (1-SF)*Smax
    # I used 2% safety factor
    Smin = 2.31 + 11.4 * Beta - 27.1 * Beta ** 2 + 16.5 * Beta ** 3
    if Beta > 0.3:
        Smax = 28.4 - 85.1 * Beta + 107 * Beta ** 2 - 46.9 * Beta ** 3
    else:
        Smax = 1.86 * Beta + 4.46
    return (1 + SF) * Smin, (1 - SF) * Smax


# Generating the input and outpot for NN
# input is called Input_NN
# output is called Output_NNt_NN
# The size of input: row=Beta_num*Smax_num column=2*num_point_integration


# initial condition
num_point_integration = 200
noise_Percent_of_datamean = 0.01

Beta_min = 0.1
Beta_max = 0.8
Beta_num = 11  # number of datapoint for Beta
Smax_num = 11  # number of datapoint for Smax

# We decide how many outputs and input we need
Num_Output_Var = 1
Num_Input_Var = 5

# size of input
Input_XGbooost = np.zeros((Beta_num * Smax_num, Num_Input_Var))
Output_XGbooost = np.zeros((Beta_num * Smax_num, Num_Output_Var))

Row_num = 0

for Beta in np.linspace(Beta_min, Beta_max, num=Beta_num):
    Smin, Smax = find_Smin_Smax(Beta, SF=0.02)
    for S in np.linspace(Smin, Smax, num=Smax_num):
        # solve ODE to generate the drop profile datapoint
        z0 = [0.0000001, 0.0000001, 0.0000001]
        t = np.linspace(0, S, num_point_integration)
        z = odeint(Drop_Profil, z0, t, args=(Beta,))  # half of drop profile
        # Data_point=np.concatenate((z[:,0:2],np.stack((-z[:,0],z[:,1]),axis=1))) #Complete drop profile

        # Adding noise to the drop profile datapoint
        x, y = Add_Noise_Drop_Profile(z, noise_Percent_of_datamean)

        # Generating outputs
        Drop_Height = y[-1]  # Drop Height
        R_Cap = x[-1]  # Capillary radius
        # plt.plot(x,y,'bo')
        R_e, R_s = Find_Re_Rs(x, y, 5, Drop_Height)  # Equatorial Radius and Rs

        Input_XGbooost[Row_num, :] = [Drop_Height, R_Cap, R_s, R_e, S]
        Output_XGbooost[Row_num, :] = Beta

        Row_num += 1

labels = ['Drop Height', 'Capillary Radius', 'R-s', 'R-e', 'Smax']
label_y = ['Beta']
df_y = pd.DataFrame(data=Output_XGbooost, columns=label_y)
df = pd.DataFrame(data=Input_XGbooost, columns=labels)

df = pd.concat([df, df_y], axis=1)
df = df.drop('Smax', axis=1)
df.to_csv('pdt-training-set.csv', index=False)

