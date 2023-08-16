"""
pdt-extract.py

- Author: Dmitri Lyalikov
- Email:  Dlyalikov01@manhattan.edu
- Date of last revision: 05/02/2023
- Status: in development / validating

This module is the entry point and controller of the profile and feature extraction sequence
implemented by the pdt-extract project.
It can be used as a standalone script or imported with DropProfile class to:
    - process one or more image files from the folder: path, or from a single .png file or image (ndarray)
    - output the extracted canny generated drop profile to the subdirectory: dest
    - extract and generate a .csv file of characteristic features to file: Feature Sets/feature_set.csv

"""

import imageio
import os
from scipy import ndimage
from feature_extract import FeatureExtract

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import pandas as pd


class DropProfile:
    def __init__(self, path: str = "../Pendant Drops", dest: str = "Drop Profiles", feature_set: str = "features.csv"):
        """
        :param path: poth to directory to access input images.
        :param dest: path to subdir to save output images. It is assumed destination dir is a subdirectory in path: (path/dest)
        :param feature_set: file name to save feature set as csv to (should include .csv)
        """
        self.path = path
        self.destination = dest
        self.feature_set = feature_set
        self.max_height = 0
        self.max_width = 0
        self.feature_list = []

    # Perform bulk profile and feature extraction on all files in self.path
    # generate drop profile .jpg and save to self.destination
    def extract_from_dir(self, canny_done=False, extract=True):
        os.chdir("../pdt_extract")
        os.chdir(self.path)
        coord_list = {}
        if canny_done:
            for filename in os.listdir():
                if not os.path.isdir(filename):
                    print(f"Extracting profile from: {filename}...")
                    profile = load_edge(filename)
                    os.chdir(self.destination)
                    image, features, x, y = self.get_profile(profile, filename, extract=extract)
                    coord_list[f"{filename}"] = [x, y]
                    os.chdir("..")
                else:
                    print(f"not file: {filename}")
            if not os.path.exists("Feature Sets"):
                os.mkdir("Feature Sets")
            df = pd.DataFrame(self.feature_list)
            df.to_csv("Feature Sets" + '/' + self.feature_set, index=False)
            os.chdir("../pdt_extract")
        else:
            for filename in os.listdir():
                if not os.path.isdir(filename):
                    print(f"Extracting profile from: {filename}...")
                    profile = extract_profile_from_image(os.path.join(filename))
                    os.chdir(self.destination)
                    image, features, x, y = self.get_profile(profile, filename, extract=extract)
                    coord_list[f"{filename}"] = [x, y]
                    os.chdir("..")
                else:
                    print(f"not file: {filename}")
            if not os.path.exists("Feature Sets"):
                os.mkdir("Feature Sets")
            df = pd.DataFrame(self.feature_list)
            df.to_csv("Feature Sets" + '/' + self.feature_set, index=False)
            os.chdir("../pdt_extract")

        print(f"Done Extracting Profiles")
        return coord_list

    # perform extraction of profile and feature set given a path to an image with respect to self.path
    def extract_from_file(self, fname: str, canny_done: bool, extract=True) -> (ndimage, list):
        if canny_done:
            profile = load_edge(fname)
            self.get_profile(profile, fname)
        else:
            os.chdir(self.path)
            profile = extract_profile_from_image(os.path.join(fname))
            return self.get_profile(profile, extract=extract)

    # perform extraction of profile and feature set given a ndimage
    def extract_from_img(self, img: ndimage, extract=True) -> (ndimage, list):
        profile = extract_profile_from_image(img, load=False, path_to_file=None)

    # label connected components as edge profiles
    def get_profile(self, final_image, filename=None, save=True, extract=True):
        labeled_image, num_features = ndimage.label(final_image)
        # show_image(labeled_image)
        # Remove feature 2 which is the internal noise from light

        final_image[labeled_image == 1] = 255

        final_image = split_profile(final_image)

        # Create ordered set of X and Y coordinates along edge profile
        indices = np.where(final_image > 0)
        x = np.flip(indices[1])
        y = np.flip(indices[0])
        reconstruct(x, y)
        if save:
            imageio.imwrite(filename, np.uint8(final_image))
        # Extract and save profile features to feature list
        if extract:
            features = FeatureExtract(x, y)
            features.feature_set["image"] = filename
            self.feature_list.append(features.feature_set)
            show_image(final_image)
            print(f"{filename}: {features.show_features()}")
            return final_image, features.feature_set, x, y
        fft_profile(final_image)

        return final_image, None, x, y


def reconstruct(x_coords, y_coords):
    image_array = np.zeros((500, 500), dtype=np.uint8)
    for x, y in zip(x_coords, y_coords):
        image_array[y, x] = 255
    show_image(image_array)

#    Execute the Canny Sequence on the image
#    gaussian_blur_sigma value = 1.2
#    high_threshold_ratio = 0.2
#    low_threshold_ratio = 0.15
def extract_profile_from_image(path_to_file: str, img: ndimage = None, load=True, extract=True):
    if load:
        img = load_convert_image(path_to_file)
    dx = ndimage.sobel(img, axis=1)  # horizontal derivative
    dy = ndimage.sobel(img, axis=0)  # vertical derivative
    mag = normalize(np.hypot(dx, dy))
    gradient = np.degrees(np.arctan2(dy, dx))
    nms = normalize(nms_with_interpol(mag, gradient, dx, dy))
    profile = hysteresis_threshold(nms)
    show_image(profile)
    return profile


# We have a grayscale ndarray.
# We want to find the vertically-lowest pixel that has the value 255.
# When we find that column, before cutting the image and keeping the right side,
# we need to make sure it is either the only vertical minimum,
# or find the midpoint between the furthest away vertical minimum column and split the image at that midpoint instead
def split_profile(img: ndimage):
    show_image(img)

    nonzero_coords = np.argwhere(img != 0)
    lowest_position = np.min(nonzero_coords, axis=0)
    # Find the indices of all pixels with value 255 along the vertical axis
    indices = np.where(img > 0)[1]

    # Find the lowest index, which corresponds to the lowest pixel in the image with value 255
    lowest_index = np.min(indices)

    # Find the columns that have this lowest pixel value
    cols = np.where(img[lowest_index, :] > 0)[0]

    # If there is only one such column, use it as the cutting point
    if len(cols) == 1:
        cutting_point = cols[0]

    # Otherwise, find the midpoint between the furthest away vertical minimum columns
    else:
        left_col = cols[0]
        right_col = cols[-1]
        midpoint = (left_col + right_col) // 2
        cutting_point = midpoint

    # Cut the image and keep the right side
    return img[:, cutting_point:]


def show_image(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


# load a preprocessed edge profile
# img: passed in as full directory
def load_edge(img: str) -> ndimage:
    lion = imageio.v2.imread(img, None)
    show_image(lion)
    # Convert to grayscale
    img = np.dot(lion[..., :3], [0.299, 0.587, 0.114])
    show_image(img)
    return img


# Load the next image in subdir
# img: passed in as full directory
def load_convert_image(img: str, sigma_val=1):
    lion = imageio.v2.imread(img, None)
    lion_gray = np.dot(lion[..., :3], [0.299, 0.587, 0.114])
    # Find the middle row index
    # Optionally change or take parameter for sigma
    img = ndimage.gaussian_filter(lion_gray, sigma=sigma_val)
    return img


# Normalize the pixel array, so that values are <= 1
def normalize(img):
    img = img / np.max(img)
    return img


# Do Non-Maximum Suppression with interpolation to get a better
# Estimate of the magnitude values of the pixels in the gradient
# Direction. This is done to get thin edges
def nms_with_interpol(g_mag, grad, gx, gy):
    nms = np.zeros(g_mag.shape)

    for i in range(1, int(g_mag.shape[0]) - 1):
        for j in range(1, int(g_mag.shape[1]) - 1):
            if grad[i, j] >= 0 and grad[i, j] <= 45 or grad[i, j] < -135 and grad[i, j] >= -180:
                y_bot = np.array([g_mag[i, j + 1], g_mag[i + 1, j + 1]])
                y_top = np.array([g_mag[i, j - 1], g_mag[i - 1, j - 1]])
                x_est = np.absolute(gy[i, j] / g_mag[i, j])
                if (g_mag[i, j] >= ((y_bot[1] - y_bot[0]) * x_est + y_bot[0]) and g_mag[i, j] >= (
                        (y_top[1] - y_top[0]) * x_est + y_top[0])):
                    nms[i, j] = g_mag[i, j]
                else:
                    nms[i, j] = 0
            if grad[i, j] > 45 and grad[i, j] <= 90 or grad[i, j] < -90 and grad[i, j] >= -135:
                y_bot = np.array([g_mag[i + 1, j], g_mag[i + 1, j + 1]])
                y_top = np.array([g_mag[i - 1, j], g_mag[i - 1, j - 1]])
                x_est = np.absolute(gx[i, j] / g_mag[i, j])
                if (g_mag[i, j] >= ((y_bot[1] - y_bot[0]) * x_est + y_bot[0]) and g_mag[i, j] >= (
                        (y_top[1] - y_top[0]) * x_est + y_top[0])):
                    nms[i, j] = g_mag[i, j]
                else:
                    nms[i, j] = 0
            if grad[i, j] > 90 and grad[i, j] <= 135 or grad[i, j] < -45 and grad[i, j] >= -90:
                y_bot = np.array([g_mag[i + 1, j], g_mag[i + 1, j - 1]])
                y_top = np.array([g_mag[i - 1, j], g_mag[i - 1, j + 1]])
                x_est = np.absolute(gx[i, j] / g_mag[i, j])
                if (g_mag[i, j] >= ((y_bot[1] - y_bot[0]) * x_est + y_bot[0]) and g_mag[i, j] >= (
                        (y_top[1] - y_top[0]) * x_est + y_top[0])):
                    nms[i, j] = g_mag[i, j]
                else:
                    nms[i, j] = 0
            if grad[i, j] > 135 and grad[i, j] <= 180 or grad[i, j] < 0 and grad[i, j] >= -45:
                y_bot = np.array([g_mag[i, j - 1], g_mag[i + 1, j - 1]])
                y_top = np.array([g_mag[i, j + 1], g_mag[i - 1, j + 1]])
                x_est = np.absolute(gy[i, j] / g_mag[i, j])
                if (g_mag[i, j] >= ((y_bot[1] - y_bot[0]) * x_est + y_bot[0]) and g_mag[i, j] >= (
                        (y_top[1] - y_top[0]) * x_est + y_top[0])):
                    nms[i, j] = g_mag[i, j]
                else:
                    nms[i, j] = 0

    return nms


# Double threshold Hysteresis
def hysteresis_threshold(img, high_threshold_ratio=0.2, low_threshold_ratio=0.15):
    high_threshold_ratio = 0.4
    low_threshold_ratio = 0.15
    g_sup = np.copy(img)
    h = int(g_sup.shape[0])
    w = int(g_sup.shape[1])
    high_threshold = np.max(g_sup) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    x = 0.1
    old_x = 0

    # The while loop is used so that the loop will keep executing till the number of strong edges
    # do not change, i.e. all weak edges connected to strong edges have been found
    while old_x != x:
        old_x = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if g_sup[i, j] > high_threshold:
                    g_sup[i, j] = 1
                elif g_sup[i, j] < low_threshold:
                    g_sup[i, j] = 0
                else:
                    if ((g_sup[i - 1, j - 1] > high_threshold) or
                            (g_sup[i - 1, j] > high_threshold) or
                            (g_sup[i - 1, j + 1] > high_threshold) or
                            (g_sup[i, j - 1] > high_threshold) or
                            (g_sup[i, j + 1] > high_threshold) or
                            (g_sup[i + 1, j - 1] > high_threshold) or
                            (g_sup[i + 1, j] > high_threshold) or
                            (g_sup[i + 1, j + 1] > high_threshold)):
                        g_sup[i, j] = 1
        x = np.sum(g_sup == 1)

    # This is done to remove/clean all the weak edges which are not connected to strong edges
    g_sup = (g_sup == 1) * g_sup

    return g_sup


# Remove connected edges that are noise
# Assuming edge profile is the longest edge
def extract_profile(img):
    labeled_image, num_features = ndimage.label(img)
    # Remove all features that are not labeled 1 or 0, (profile or background)
    img[labeled_image == 2] = 0
    img[labeled_image == 1] = 255
    return img



# Fast Fourier Transform of edge profile
# Can expect high frequency components in magnitude spectrum of edges
# Computed in Decibels
def fft_profile(profile):
    fft_image = fft.fft2(profile)
    fft_image = fft.fftshift(fft_image)
    # Shift the zero-frequency component to the center of the spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fft_image))
    phase_spectrum = np.angle(fft_image)


if __name__ == '__main__':
    profiles = DropProfile()
    profiles.extract_from_dir(canny_done=False, extract=False)
    # profiles.extract_from_file(fname="../matlab_canny/d-1-55.png", canny_done=True)
