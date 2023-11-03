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
from circle_fit import taubinSVD, lm, prattSVD, hyperSVD
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import cv2


class FeatureExtract:
    def __init__(self, x: list[int], y: list[int], tensor_file="tensors.csv"):
        """
        :param x: globally used ordered set of x coordinates of the pendant drop profile
        :param y: globally used ordered set of x coordinates of the pendant drop profile
        """

        self.x = x
        self.y = y
        self.tensor_file = tensor_file

        self.feature_positions = {
            "Drop Height": [1, 0, 0],
            "Capillary": [0, -1, -1],
            "Apex": [0, 0, 0],
            "Equator Radius": [0, 0, 0],
            "R-s": [0, 0, 0]
        }

        self.capillary_radius = self.x[-1]
        self.drop_height = self.y[0]
        self.equator_radius, self.s_radius = self.find_re_rs()

        self.apex_radius = 172# self.find_apex_radius()
        self.build_tensor()
        # Normalize to dimensionless ratio to apex radius
        self.feature_set = {
            "Drop height": self.drop_height / self.apex_radius,
            "Capillary radius": self.capillary_radius / self.apex_radius,
            "R-s": self.s_radius / self.apex_radius,
            "R-e": self.equator_radius / self.apex_radius,
            "Apex Radius": self.apex_radius
        }
        self.xgb_beta = self.predict_beta()
        self.feature_set["XGBoost Beta"] = self.xgb_beta
        # self.feature_set["LightGBM Beta"] = self.lgbm_beta
        """
        print(f"Apex radius (Pixels): {self.apex_radius }")
        print(f"Equator radius: {self.equator_radius }\n"
              f"S radius: {self.s_radius }\n"
              f"Capillary radius: {self.capillary_radius}\n"
              f"Drop Height: {self.drop_height }")
        """
        # rgi = ProfileGenerator(self.x, self.y, features=self.feature_positions)

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
            self.feature_positions["Equator Radius"][1] += 1
            self.feature_positions["Equator Radius"][2] += 1
            output = self.recursive_equator_radius(i, n)
            if output is not None:
                self.equator_radius = output
                # self.feature_positions["R-e"] = (0, i, i)
        else:
            if i <= len(self.x) * 0.7:
                # assumed 70% of drop is enough for the equator radius
                self.feature_positions["Equator Radius"] = (0, i, i)
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
            self.feature_positions["Equator Radius"] = (0, xc, yc)

        # Find s_radius at y = 2 * equator_radius
        if self.equator_radius < 0.5 * self.drop_height:
            # res = index of y if y > 2 * equator_radius
            res = next(xx for xx, val in enumerate(np.flip(self.y)) if val > 2 * self.equator_radius)
            self.s_radius = self.x[res]
            self.feature_positions["R-s"] = (0, res, res)
        else:
            # Drop is too small
            self.s_radius = self.capillary_radius
            self.feature_positions["R-s"] = self.feature_positions["Capillary"]
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
        self.feature_positions["Apex"] = (0, len(r_0), len(r_0))
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

    def predict_beta(self):
        input_series = []
        for key, value in self.feature_set.items():
            input_series.append(value)
        input_series.pop(-1)
        input_array = np.array(input_series).reshape(1, -1)
        xgb = pickle.load(open('../../models/xgboost-8-27.pkl', "rb"))
        lgbm = pickle.load(open('../../models/8-27-lightgbm.pkl', "rb"))

        xgb_prediction = xgb.predict(input_array)[0]
        # lgbm_prediction = lgbm.predict(input_array)

        return xgb_prediction  # , lgbm_prediction


    def show_image(self, img):
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()


    """
    def build_tensor(self):
        Take X and Y list and construct normalized tensor of size 400
        Format: x0, y0, x1, y1, ... x199, y199 
        Where x0, y0 is the first point in the profile and all points are normalized to the Apex Radius
        Write as row to csv file: self.tensorfile
        :return: tensor of size 200 
    """
    def build_tensor(self):
        tensor = []
        for i in range(len(self.x)):
            tensor.append(self.x[i] / self.apex_radius)
            tensor.append(self.y[i] / self.apex_radius)
        # If tensor is less than 400 points, pad with 0s at end

        tensor = pad_truncate(tensor)

        print(len(tensor))

        # Convert to dataframe and write to csv
        tensor = np.array(tensor)
        tensor = tensor.reshape(1, -1)
        df = pd.DataFrame(tensor)
        df.to_csv(".." + '/' + "tensors" + '/' + self.tensor_file, mode='a', header=False, index=False)

        return tensor



"""
    Pad or truncate a list to have exactly 400 points.

    If the input list `tensor` has more than 400 points, it removes evenly spaced points across the entire list to
    reduce it to 400 points. If it has fewer than 400 points, it pads the list with zeros to reach the desired length.

    Parameters:
    tensor (list): A list of data points.

    Returns:
    list: The modified list with exactly 400 points.
"""
def pad_truncate(tensor: list):
    # If tensor is greater than 400 points, remove evenly spaced points across whole tensor so that it is 400 points
    print(tensor)
    num_pairs = len(tensor) // 2
    if len(tensor) > 400:
        target_pairs = 200
        step = (num_pairs // (target_pairs // 2)) + 1

        while num_pairs > target_pairs:
            indices_to_remove = [i for i in range(step - 1, num_pairs, step)]
            indices_to_remove.reverse()
            print(indices_to_remove)
            # Remove elements in reverse order to minimize impact on the end of the list
            for index in indices_to_remove:
                del tensor[2 * index:2 * (index + 1)]
            num_pairs = len(tensor) // 2

    if len(tensor) < 400:
        for i in range(400 - len(tensor)):
            tensor.append(0)

    return tensor

class ProfileGenerator:
    def __init__(self, x, y, features, height=500, width=500):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.feature_positions = features
        self.image = np.zeros((height, width), dtype=np.uint8)
        for x, y in zip(self.x, self.y):
            self.image[y, x] = 255
        self.generate_image()
        self.show_image(self.image)

    def generate_image(self):
        for feature, (direction, x, y) in self.feature_positions.items():
            self.fill_line(x, y, direction, self.width if direction == 0 else self.height)
            if feature != "Drop Height":
                self.add_text(self.x[x], self.y[y], feature)  # Add feature label above the line

    def add_text(self, x, y, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y - text_size[1] - 5
        cv2.putText(self.image, text, (text_x, text_y), font, font_scale, font_color, thickness)

    def fill_line(self, x, y, direction, length):
        if direction == 0:
            self.image[self.y[y], :self.x[x] + 1] = 255
        elif direction == 1:
            self.image[:self.y[y] + 1, self.x[x]] = 255

    def show_image(self, img):
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()

def Find_Re_Rs(x,y,n,Drop_Height, R_Cap):
  rs_pos, re_pos = 0, 0
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
        re_pos = i
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
    re_pos = xc
  # Find R_s at y=2*R_e
  if R_e<0.5*Drop_Height:
    #res=index of y if y>2*R_e
    res = next(xx for xx, val in enumerate(y) if val > 2*R_e)
    R_s=x[res]
    rs_pos = res

  else:
    # Drop is too small
    R_s=R_Cap
    rs_pos = 0
  return R_e,R_s, re_pos, rs_pos



