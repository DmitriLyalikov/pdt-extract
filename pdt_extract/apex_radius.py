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

def split_lists_by_percent(list1, list2, percent):
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length")

    if percent < 0 or percent > 50:
        raise ValueError("Percent should be between 0 and 50")

    middle_index = len(list1) // 2
    middle_index = 0
    offset = int(len(list1) * percent / 100)

    left_list1 = list1[middle_index - offset : middle_index]
    right_list1 = list1[middle_index : middle_index + offset + 1]

    left_list2 = list2[middle_index - offset : middle_index ]
    right_list2 = list2[middle_index : middle_index + offset + 1]

    combined_list1 = left_list1 + right_list1
    combined_list2 = left_list2 + right_list2

    return combined_list1, combined_list2
    #return right_list1, right_list2

def build_plot(x, y):
    plt.plot(x, y)
    plt.xlabel('Percent')
    plt.ylabel('Apex Radius')
    plt.title(' Percent Drop on Apex Radius')

    plt.show()


def sort_and_reshuffle_lists(first_list, second_list):
    midpoint = len(first_list) // 2

    elements_with_indices = [(element, idx, 0 if idx < midpoint else 1) for idx, element in enumerate(first_list)]


    sorted_elements = sorted(elements_with_indices, key=lambda x: (x[0], -x[2]))

    sorted_indices = [idx for _, idx, _ in sorted_elements]

    reshuffled_second_list = [second_list[idx] for idx in sorted_indices]

    return sorted_indices, reshuffled_second_list

class ApexBuilder:
    def __init__(self, x: list[int], y: list[int]):
        """
        :param x: globally used ordered set of x coordinates of the pendant drop profile
        :param y: globally used ordered set of x coordinates of the pendant drop profile
        """

        self.x = x
        self.y = y


        indexed_y = list(enumerate(self.y))
        indexed_x = list(enumerate(self.x))
        y_indices = []
        y_rearranged = []
        # Sort the indexed list by the values
        sorted_y = sorted(indexed_y, key=lambda x: x[1])
        for index, value in sorted_y:
            print(f"Index: {index}, Value: {value}")
            y_indices.append(index)
            y_rearranged.append(value)


        x_rearranged = [None] * len(self.x)


        # Rearrange the elements based on the index list
        for i, index in enumerate(y_indices):
            x_rearranged[index] = x[i]
        #print(x)
        #print(y)
        #print(x_rearranged)

        #sorted_x = sorted(indexed_x, key=lambda x: x[1])
        #for index, value in sorted_x:
        #    print(f"Index: {index}, Value: {value}")


        percents = list(range(1, 51))
        apex_radii = []
        for percent_drop in percents:
            y, x = split_lists_by_percent(y_rearranged, x_rearranged, percent_drop)
            apex_radius = find_apex_radius(x, y)
            apex_radii.append(apex_radius)
            print(f"Percent from middle: {percent_drop}, Apex Radius: {apex_radius}")
        build_plot(percents, apex_radii)
        """
        Find middle of each list
        make micro list that is p% of macro list on both sides and extract values
        pass into apex radius function
        increment p, and plot apex radius vs p
        tune percent_drop_r0
        """

        # Normalize to dimensionless ratio to apex radius




    def show_features(self):
        str_features = ""
        for key, value in self.feature_set.items():
            str_features += key + " " + str(value) + " "
        return str_features


    # Use Circle fit to approximate apex radius of edge profile
    # ratio_drop_length: 1 >= float value > 0 representing number points along profile to approximate with
    # change_ro: float value representing minimum value of change in circle radius before stopping approximation
def find_apex_radius(x, y, ratio_drop_length: float = 0.2, change_ro: float = 2) -> float:

    num_point_ro_circlefit = round(len(x) * ratio_drop_length) + 1

    percent_drop_ro = 1
    i = 0
    diff = 0
    r0 = 0
    r_0 = []
    while diff >= change_ro*r0 or num_point_ro_circlefit <= percent_drop_ro * len(x):
        points_ro_circlefit = np.stack((x[:num_point_ro_circlefit], y[:num_point_ro_circlefit]), axis=1)
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


def split_lists_by_percent(list1, list2, percent):
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length")

    if percent < 0 or percent > 50:
        raise ValueError("Percent should be between 0 and 50")

    middle_index = len(list1) // 2
    offset = int(len(list1) * percent / 100)

    left_list1 = list1[middle_index - offset : middle_index]
    right_list1 = list1[middle_index : middle_index + offset + 1]

    left_list2 = list2[middle_index - offset : middle_index ]
    right_list2 = list2[middle_index : middle_index + offset + 1]

    combined_list1 = left_list1 + right_list1
    combined_list2 = left_list2 + right_list2

    return combined_list1, combined_list2

# Example usage:
list1 = [10, 5, 6, 5, 2, 9, 1]
list2 = [20, 15, 16, 12, 10, 25, 5]
percent = 15.2

combined_list1, combined_list2 = split_lists_by_percent(list1, list2, percent)
print("Combined list 1:", combined_list1)
print("Combined list 2:", combined_list2)

