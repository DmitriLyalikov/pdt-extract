# Canny-Edge-Detector
This module will process all image files from the folder: pendant_drops
and output the extracted canny generated drop profile to the subdirectory: drop_profiles

## Table of Contents
1. [Usage](#usage)
2. [Profile Extraction](#Profile-Extraction)
3. [Feature Extraction](#Feature-Extraction)
4. [Results](#Results)
5. [Appendix](#appendix)

## Usage
* Download and extract .zip file or:
```
git clone https://github.com/DmitriLyalikov/Canny-Edge-Detector-master.git
```
Navigate to root directory where this project was stored locally

* Upload Pendant Drop Images as PNG to the Pendant Drop subdirectory
This tool currently only works with .jpg images 

```
pip install -e .
python pdt-canny.py
```
This will output the extracted profiles to the subdirectory Drop Profiles for each image

### Usage as a package
To use this package else where:
```pycon
pip install pdt_extract
```
Create a new DropProfile class and generate profiles and feature sets from images in a directory:
```python
from pdt_extract import DropProfile

profiles = DropProfile(path="path/to/pendant/drop/images", dest="path/to/save/to")
# Automated extraction from image directory
profiles.extract_from_dir()

# extract and save from one .png file
profiles.extract_from_file("image_name.png")

# Extract from a numpy nd.image and return as python objects for further processing
profiles.extract_from_img(image_as_ndimage)
```

```python
from pdt_extract.pdt_extract import DropProfile
```
## Profile Extraction
Given a raw image of a pendant drop, the profile is extracted through a series of steps, including the canny edge detection sequence, removing reflective noise,
and splitting the image at its apex.
### Smoothing (Gaussian blur)
To reduce the image noise, a guassian filter is applied to every pixel in the image.
The function acts as a filter, blurring the edges and reducing the contrast between adjacent pixels.
The degree of blurring is controlled by the standard deviation (sigma). A larger sigma results in more blue

* Currently, the best sigma values for the sample images are 1.2-1.4, this will depend on each image

### Gradient Calculation (Sobel Filter)
The sobel filter computes the gradient of the image intensity at each pixel. This 
is a measure of how quickly the intensity of the image changes at that point, using convolution kernel

The convolution kernels are applied in both horizontal and vertical direction. Each pixel in the output of the Sobel
represents the gradient magnitude at that point. 

### Normalize 
Normalizing an image means transforming the pixel values of an image to a common scale or range of values, 
typically between 0 and 1 or between -1 and 1. This is done to improve the accuracy and reliability of various image processing tasks, 
such as image classification, object detection, and segmentation.

The process of normalization involves subtracting the mean pixel value from each pixel in the image and then dividing by the standard deviation 
of the pixel values. This has the effect of centering the pixel values around zero and scaling them to have unit variance.

### Calculate the Euclidean Magnitude (np.hypot())
After normalizing our sobel filtered image, we have a gradient map.
The np.hypot() function will compute the euclidean magnitude and return
an image of corresponding magnitude of the 2d vector. 

### Non-Maxima Suppression 
A non-maxima suppression fucntion is used to thin out the edges, by reducing them 
to a single pixel. 

* Compute the gradient magnitude and orientation at each pixel,
* for each pixel on the edge, compare the magnitude of its gradient with magnitudes
of its neighbors along the gradient direction
* If the magnitude is greater than the magnitude of its two neighbors, it is retained as an edge

## Double Hysterisis Thresholding
Two thresholds are used to classify pixels as strong, weak, or non edges. The thresholds
are typically chosesn to be a high threshold and a low threshold, where the high threshold is greater.

* Apply the high threshold to the gradient magnitude image to identify strong edge pixels. 
Pixels with magnitudes greater than or equal to the high threshold are classified as strong edge pixels.
* Apply the low threshold to the gradient magnitude image to identify weak edge pixels. 
Pixels with magnitudes less than the high threshold but greater than or equal to the low threshold are classified as weak edge pixels
* Any pixels with magnitudes less than the low threshold are classified as non-edge pixels.
* Connect weak edge pixels to strong edge pixels if they are adjacent to each other in the image. 
This is typically done by tracing a path along the chain of weak edges until a strong edge is encountered.
* The resulting image consists of only strong edge pixels and weak edge pixels that are connected to strong edge pixels.


## Feature Extraction
### Circle Fit

Please refer to this excellent link to better understand the algorithm : "http://justin-liang.com/tutorials/canny/"

Important Points:
- I have used a relatively slow iterative approach to perform the function of Double Thresholding Hysterisis,
  a better and time-saving alternative is to use a recursive algorithm which tracks the edges.
- The value of Sigma to implement Gaussian Blur is image specific, different values can be tested to see which give the best estimate of edges.
- The ratio of the thresholds is again another variable, but the ones that I have used in the code give pretty good estimates for any particular image.
- Non Maxima Suppression with Interpolation although being computationally expensive, gives excellent estimates and is better tha NMS without interpolation
