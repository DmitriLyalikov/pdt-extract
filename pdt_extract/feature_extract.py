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
        print(x[0])
        print(y[0])
        print(x[-1])
        print(y[-1])
        self.capillary_radius = self.x[-1]
        self.drop_height = self.y[0]
        self.equator_radius, self.s_radius = self.find_re_rs(3)
        self.apex_radius = self.find_apex_radius()
        self.feature_set = {
            "Apex radius": self.apex_radius,
            "Equator radius": self.equator_radius,
            "S_radius": self.s_radius,
            "Capillary radius": self.capillary_radius,
            "Drop height": self.drop_height
        }

        print(f"Apex radius: {self.apex_radius * (0.05 / 44)}")
        print(f"Equator radius: {self.equator_radius * (0.05 / 44)},"
              f"S radius: {self.s_radius * (0.05 / 44)},"
              f"Capillary radius: {self.capillary_radius * (0.05 / 44)},"
              f"Drop Height: {self.drop_height * (0.05 / 44)}")
        self.equator_radius, self.s_radius = Find_Re_Rs(self.x, self.y, 5, self.drop_height)  # Equatorial Radius and Rs
        print(f"Equator radius: {self.equator_radius * (0.05 / 44)},"
              f"S radius: {self.s_radius * (0.05 / 44)}")

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
                if i <= len(self.x) * .7:
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
            # res = index of y if y > 2 * equator_radius
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

# divide all elements R0

R_e = 0

#Finding Equator radius (Re) and Rs @ y=2Re
def Find_Re_Rs(x,y,n,Drop_Height):
  # at i_th x: we average from i-n to i+n which means 11 points
  global R_e
  R_e=0
  i=n
  def Average_x(x,i,n):
      # i-n>=0
    s=0
    for j in range(i-n,i+n+1):
      s=s+x[j]
    return s/(2*n+1)
  def Recur_Equator_Radius(x,i,n):
    global R_e
  #R_e is the equator radius: must be defined here not outside
  #We use recursive approach: Start from Apex continue until x decreases
  #At i-th point we averagne x of x-n to x+n to subpress noise
  #We compare x_i_th vs x_i+1_th until it decrease to find equator
    if Average_x(x,i,n)<Average_x(x,i+1,n) and i<=len(x)-n-3:
      i=i+1
      Output=Recur_Equator_Radius(x,i,n)
      if Output is not None:
        # Since recursive returns None!!! we use global Variable
        R_e=Output
    else:
      if i<=len(x)*0.7:
        # I assumed 70% of drop is enough for R_e
        # print(i)
        # print(x[i])
        return x[i]
      else:
        return
  #A recursive function that returns equator radius
  Recur_Equator_Radius(x,i,n)
  if R_e==0:
    # R_e=0: drop is not well-deformed e.g. Beta>0.7. Find R_e from cirle fitting
    # I selected 40% of the total number of points for circle fitting
    num_point_RH_Circlefit=round(0.4*len(x))
    Points_RH_Circlefit=np.stack((x[:num_point_RH_Circlefit],y[:num_point_RH_Circlefit]),axis=1)
    xc, yc, R_e, sigma = taubinSVD(Points_RH_Circlefit)

  # Find R_s at y=2*R_e
  if R_e<0.5*Drop_Height:
    #res=index of y if y>2*R_e
    res = next(xx for xx, val in enumerate(y) if val > 2*R_e)
    R_s=x[res]
  else:
    # Drop is too small
    R_s= 0.1931818181818182
  return R_e,R_s

#Finding drop apex radius
def Find_apex_Radius(x,y,num_point_Ro_Circlefit,Change_Ro):
  # I selected 10 points from apex for circle fitting
  Percenr_Drop_Ro=0.1
  i=0
  diff=0
  R0=0
  R_o=[]
  while diff>=Change_Ro*R0 or num_point_Ro_Circlefit<=Percenr_Drop_Ro*len(x):
    Points_Ro_Circlefit=np.stack((x[:num_point_Ro_Circlefit],y[:num_point_Ro_Circlefit]),axis=1)
    xc, yc, R0, sigma = taubinSVD(Points_Ro_Circlefit)
    R_o.append(R0)
    if i>1:
      diff=abs(R_o[i]-R_o[i-1])
    i+=1
    num_point_Ro_Circlefit+=1
  # print(R_o)
  return R_o[-1]



