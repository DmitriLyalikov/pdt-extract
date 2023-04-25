"""
Author: Dmitri Lyalikov-Dlyalikov01@manhattan.edu

Canny Edge Detection Processor
This module will process all image files from the folder: pendant_drops
and output the extracted canny generated drop profile to the subdirectory: drop_profiles

TODO: Implement Feature extraction from apex radius and:
    - generate CSV of features from all profiles with extract_from_dir
    - return dictionary of features {Drop Height, Capillary Radius, Rs, Re} with extract_from_file, extract_from_img
"""

import imageio
import os
from scipy import ndimage

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from circle_fit import taubinSVD


class DropProfile:
    def __init__(self, path="Pendant Drops", dest="Drop Profiles"):
        self.path = path
        self.destination = dest
        self.max_height = 0
        self.max_width = 0

    # Perform bulk profile and feature extraction on all files in self.path
    # generate drop profile .jpg and save to self.destination
    def extract_from_dir(self):
        os.chdir(self.path)
        for filename in os.listdir():
            if not os.path.isdir(filename):
                print(f"Extracting profile from: {filename}...")
                profile = extract_profile_from_image(os.path.join(filename))
                os.chdir(self.destination)
                get_profile(profile, filename)
                os.chdir("..")
            else:
                print(f"not file: {filename}")

        print(f"Done Extracting Profiles")

    # perform extraction of profile and feature set given a path to an image with respect to self.path
    def extract_from_file(self, fname: str) -> (ndimage, list):
        os.chdir(self.path)
        profile = extract_profile_from_image(os.path.join(fname))
        return get_profile(profile)

    # perform extraction of profile and feature set given a ndimage
    def extract_from_img(self, img: ndimage) -> (ndimage, list):
        profile = extract_profile_from_image(img, load=False, path_to_file=None)


# label connected components as edge profiles
def get_profile(final_image, filename=None, save=True):
    labeled_image, num_features = ndimage.label(final_image)
    # Remove feature 2 which is the internal noise from light
    final_image[labeled_image == 2] = 0
    final_image[labeled_image == 1] = 255
    final_image = split_profile(final_image)
    R0 = find_apex_radius(final_image, 0.15, 0.005)
    print(f"Apex_Radius (cm): {R0}")  # 0.05 /44
    show_image(final_image)

    fft_profile(final_image)
    if save:
        imageio.imwrite(filename, np.uint8(final_image))
    else:
        return final_image, {"Apex Radius": R0}


# Use Circle fit to approximate apex radius of edge profile
# ratio_drop_length: 1 >= float value > 0 representing number points along profile to approximate with
# change_ro: float value representing minimum value of change in circle radius before stopping approximation
def find_apex_radius(profile: ndimage, ratio_drop_length: float, change_ro: float) -> float:
    indices = np.where(profile == 255)
    x = np.flip(indices[1])
    y = np.flip(indices[0])

    num_point_ro_circlefit = round(len(x) * ratio_drop_length) + 1

    percent_drop_ro = 0.1
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


#    Execute the Canny Sequence on the image
#    gaussian_blur_sigma value = 1.2
#    high_threshold_ratio = 0.2
#    low_threshold_ratio = 0.15
def extract_profile_from_image(path_to_file: str, img: ndimage = None, load=True):
    if load:
        img = load_convert_image(path_to_file)
    dx = ndimage.sobel(img, axis=1)  # horizontal derivative
    dy = ndimage.sobel(img, axis=0)  # vertical derivative
    mag = normalize(np.hypot(dx, dy))
    gradient = np.degrees(np.arctan2(dy, dx))
    nms = normalize(nms_with_interpol(mag, gradient, dx, dy))
    profile = hysteresis_threshold(nms)
    return profile


# We have a grayscale ndarray.
# We want to find the vertically-lowest pixel that has the value 255.
# When we find that column, before cutting the image and keeping the right side,
# we need to make sure it is either the only vertical minimum,
# or find the midpoint between the furthest away vertical minimum column and split the image at that midpoint instead
def split_profile(img: ndimage):
    # Find the indices of all pixels with value 255 along the vertical axis
    indices = np.where(img == 255)[0]

    # Find the lowest index, which corresponds to the lowest pixel in the image with value 255
    lowest_index = np.min(indices)

    # Find the columns that have this lowest pixel value
    cols = np.where(img[lowest_index, :] == 255)[0]

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


# Load the next image in subdir
# img: passed in as full directory
def load_convert_image(img: str, sigma_val=1.2):
    lion = imageio.v2.imread(img, None)
    lion_gray = np.dot(lion[..., :3], [0.299, 0.587, 0.114])
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
    high_threshold_ratio = 0.2
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
    profiles.extract_from_dir()
