"""
Author: Dmitri Lyalikov-Dlyalikov01@manhattan.edu

Canny Edge Detection Processing Script
This script will process all image files from the folder: pendant_drops
and output the extracted drop profile to the subdirectory: drop_profiles
"""

import imageio
import os
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps


class DropProfile:
    def __init__(self, path="pendant_drops"):
        self.path = path
        self.destination = "drop_profiles"

    def extract_from_dir(self):
        print(os.getcwd())
        os.chdir(self.path)
        for filename in os.listdir():
            if os.path.isdir(filename) == False:
                print(f"Extracting profile from: {filename}...")
                profile = extract_profile_from_image(os.path.join(filename))
                os.chdir(self.destination)
                get_profile(profile, filename)
                os.chdir("..")
            else:
                print(f"not file: {filename}")

        print(f"Done Extracting Profiles")

# label connected components as edge profiles
def get_profile(final_image, filename):
    labeled_image, num_features = ndimage.label(final_image)
    # Remove feature 2 which is the internal noise from light
    final_image[labeled_image == 2] = 0
    final_image[labeled_image == 1] = 255
    plt.imshow(final_image, cmap = plt.get_cmap('gray'))
    plt.show()
    imageio.imwrite(filename, np.uint8(final_image))
def extract_profile_from_image(image):
    img = load_convert_image(image)
    gx = normalize(sobel_filter(img, 'x'))
    gy = normalize(sobel_filter(img, 'y'))
    dx = ndimage.sobel(img, axis=1)  # horizontal derivative
    dy = ndimage.sobel(img, axis=0)  # vertical derivative
    Mag = normalize(np.hypot(dx, dy))
    Gradient = np.degrees(np.arctan2(dy, dx))
    NMS = normalize(nms_with_interpol(Mag, Gradient, dx, dy))
    fa = hysterisis_threshold(NMS)
    mag = normalize(np.hypot(gx, gy))
    gradient = np.degrees(np.arctan2(gy, gx))
    nms = normalize(nms_with_interpol(mag, gradient, gx, gy))
    img = hysterisis_threshold(nms)
    return fa


def show_image(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


# Load the next image in subdir
# img: passed in as full directory
def load_convert_image(img: str, sigma_val=1.2):
    lion = imageio.v2.imread(img, None)
    lion_gray = np.dot(lion[...,:3], [0.299, 0.587, 0.114])
    # Optionally change or take parameter for sigma
    img = ndimage.gaussian_filter(lion_gray, sigma=sigma_val)
    return img


# Apply Sobel Filter using convolution operation
# Note that in this case I have used the filter to have a maximum magnitude of 2
# It can also be changed to other numbers for aggressive edge extraction
# For eg [-1, 0, 1], [-5, 0, 5], [-1, 0, 1]
def sobel_filter(img, direction):
    if (direction == 'x'):
        Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        Res = ndimage.convolve(img, Gx)
        # Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if (direction == 'y'):
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
        Res = ndimage.convolve(img, Gy)
        # Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res

# Normalize the pixel array, so that values are <= 1
def normalize(img):
    # img = np.multiply(img, 255 / np.max(img))
    img = img / np.max(img)
    return img


# Do Non-Maximum Suppression with interpolation to get a better
# Estimate of the magnitude values of the pixels in the gradient
# Direction. This is done to get thin edges
def nms_with_interpol(Gmag, Grad, Gx, Gy):
    NMS = np.zeros(Gmag.shape)

    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if ((Grad[i, j] >= 0 and Grad[i, j] <= 45) or (Grad[i, j] < -135 and Grad[i, j] >= -180)):
                yBot = np.array([Gmag[i, j + 1], Gmag[i + 1, j + 1]])
                yTop = np.array([Gmag[i, j - 1], Gmag[i - 1, j - 1]])
                x_est = np.absolute(Gy[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if ((Grad[i, j] > 45 and Grad[i, j] <= 90) or (Grad[i, j] < -90 and Grad[i, j] >= -135)):
                yBot = np.array([Gmag[i + 1, j], Gmag[i + 1, j + 1]])
                yTop = np.array([Gmag[i - 1, j], Gmag[i - 1, j - 1]])
                x_est = np.absolute(Gx[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if ((Grad[i, j] > 90 and Grad[i, j] <= 135) or (Grad[i, j] < -45 and Grad[i, j] >= -90)):
                yBot = np.array([Gmag[i + 1, j], Gmag[i + 1, j - 1]])
                yTop = np.array([Gmag[i - 1, j], Gmag[i - 1, j + 1]])
                x_est = np.absolute(Gx[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if ((Grad[i, j] > 135 and Grad[i, j] <= 180) or (Grad[i, j] < 0 and Grad[i, j] >= -45)):
                yBot = np.array([Gmag[i, j - 1], Gmag[i + 1, j - 1]])
                yTop = np.array([Gmag[i, j + 1], Gmag[i - 1, j + 1]])
                x_est = np.absolute(Gy[i, j] / Gmag[i, j])
                if (Gmag[i, j] >= ((yBot[1] - yBot[0]) * x_est + yBot[0]) and Gmag[i, j] >= (
                        (yTop[1] - yTop[0]) * x_est + yTop[0])):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0

    return NMS


# Double threshold Hysterisis
def hysterisis_threshold(img, high_thresh_ratio=0.2, low_thresh_ratio=0.15):
    highThresholdRatio = 0.2
    lowThresholdRatio = 0.15
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    x = 0.1
    oldx = 0

    # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
    while (oldx != x):
        oldx = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (GSup[i, j] > highThreshold):
                    GSup[i, j] = 1
                elif (GSup[i, j] < lowThreshold):
                    GSup[i, j] = 0
                else:
                    if ((GSup[i - 1, j - 1] > highThreshold) or
                            (GSup[i - 1, j] > highThreshold) or
                            (GSup[i - 1, j + 1] > highThreshold) or
                            (GSup[i, j - 1] > highThreshold) or
                            (GSup[i, j + 1] > highThreshold) or
                            (GSup[i + 1, j - 1] > highThreshold) or
                            (GSup[i + 1, j] > highThreshold) or
                            (GSup[i + 1, j + 1] > highThreshold)):
                        GSup[i, j] = 1
        x = np.sum(GSup == 1)

    GSup = (GSup == 1) * GSup  # This is done to remove/clean all the weak edges which are not connected to strong edges

    return GSup


# Remove connected edges that are noise
# Assuming edge profile is the longest edge
def extract_profile(img):
    labeled_image, num_features = ndimage.label(img)
    print(num_features)
    # Remove all features that are not labeled 1 or 0, (profile or background)
    img[labeled_image == 2] = 0
    img[labeled_image == 1] = 255
    #show_image(labeled_image)
    return img


profiles = DropProfile()
profiles.extract_from_dir()