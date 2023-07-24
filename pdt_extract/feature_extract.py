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
        self.capillary_radius = self.x[-1]
        self.drop_height = self.y[0]
        self.equator_radius, self.s_radius = self.find_re_rs()
        self.equator_radius, self.s_radius = Find_Re_Rs(x[::-1], y[::-1], 15, self.drop_height, self.capillary_radius)
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

    def Find_Re_Rs(self, x, y, n):
        global R_e
        R_e = 0
        i = n


def Find_Re_Rs(x,y,n,Drop_Height, R_Cap):

  #Averaging function used in
  def Average_x(x,i,n):
  # i-n>=0. So I set i=n
    s=0
    for j in range(i-n,i+n+1):
        s=s+x[j]
    return s/(2*n+1)

  #Finding Re using updated recursive.
  def Recur_Equator_Radius(x,i,n):
  #base condition
      if Average_x(x,i,n)>Average_x(x,i+1,n):
        return x[i]
      elif i>=len(x)-n-3 or i>len(x)*0.7:
        # I searcg 70% for finding R_e
        return None
      return Recur_Equator_Radius(x,i+1,n)
  i=n
  R_e=Recur_Equator_Radius(x,i+1,n)
  if R_e is None:
  # R_e=None: drop is not well-deformed e.g. Beta>0.7. Find R_e from cirle fitting
  # I selected 30% of the total number of points for circle fitting
    num_point_RH_Circlefit=round(0.3*len(x))
    Points_RH_Circlefit=np.stack((x[:num_point_RH_Circlefit],y[:num_point_RH_Circlefit]),axis=1)
    xc, yc, R_e, sigma = taubinSVD(Points_RH_Circlefit)

  # Find R_s at y=2*R_e
  if R_e<0.5*Drop_Height:
    #res=index of y if y>2*R_e
    res = next(xx for xx, val in enumerate(y) if val > 2*R_e)
    R_s=x[res]
  else:
    # Drop is too small
    R_s=R_Cap
  return R_e,R_s

