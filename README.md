# pdt-extract
A python implementation and library for extraction of edge profiles and characteristic 
features from images of pendant drops suspended from a capillary.

## Table of Contents
1. [Usage](#usage)
2. [Profile Extraction](#Profile-Extraction)
3. [Feature Extraction](#Feature-Extraction)
4. [Results](#Results)
5. [Appendix](#appendix)

## Usage

To install this package:
```pycon
pip install pdt_extract
```
* Upload pendant drop images as PNG to the **path** subdirectory
This tool currently only works with .png images. It also assumes that the images are already cropped, removing the capillary.
* This tool also assumed the provided destination folder is a subdirectory of **path** (path/destination)
* The feature_set csv file is also assumed to be saved here: (**path**/**dest**/**name-of-csv.csv**)

See **Pendant Drops** subdirectory for valid sample images.

Import the package and create a new DropProfile class and generate profiles and feature sets from image(s).
```python
from pdt_extract import DropProfile

profiles = DropProfile(path="path/to/pendant/drop/images", dest="path/to/save/to", feature_set="name-of-csv.csv")
# Automated extraction from image directory
profiles.extract_from_dir()

# extract and save from one .png file
profiles.extract_from_file("image_name.png")

# Extract from a numpy nd.image and return as python objects for further processing
profiles.extract_from_img(image_as_ndimage)
```

## Profile Extraction
Given a raw image of a pendant drop, the profile is extracted through a series of steps, including the canny edge detection sequence, removing reflective noise,
and splitting the image at its apex.

*Example input image:*

![d-0-55.png](doc_imgs%2Fd-0-55.png)

### Smoothing (Gaussian blur)
To reduce the image noise, a Gaussian filter is applied to every pixel in the image.
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
A non-maxima suppression function is used to thin out the edges, by reducing them 
to a single pixel. 

* Compute the gradient magnitude and orientation at each pixel,
* for each pixel on the edge, compare the magnitude of its gradient with magnitudes
of its neighbors along the gradient direction
* If the magnitude is greater than the magnitude of its two neighbors, it is retained as an edge

### Double Hysteresis Thresholding
Two thresholds are used to classify pixels as strong, weak, or non edges. The thresholds
are typically chosen to be a high threshold and a low threshold, where the high threshold is greater.

* Apply the high threshold to the gradient magnitude image to identify strong edge pixels. 
Pixels with magnitudes greater than or equal to the high threshold are classified as strong edge pixels.
* Apply the low threshold to the gradient magnitude image to identify weak edge pixels. 
Pixels with magnitudes less than the high threshold but greater than or equal to the low threshold are classified as weak edge pixels
* Any pixels with magnitudes less than the low threshold are classified as non-edge pixels.
* Connect weak edge pixels to strong edge pixels if they are adjacent to each other in the image. 
This is typically done by tracing a path along the chain of weak edges until a strong edge is encountered.
* The resulting image consists of only strong edge pixels and weak edge pixels that are connected to strong edge pixels.

*Example output of canny sequence:*


![img.png](doc_imgs/img.png)

### Reflective Noise Removal
The smaller internal edge is detected from the reflected light of the pendant drop when
the image is taken. Since this edge is connected and assumed to be always smaller than the edge profile, the smaller edge is isolated and filtered out:
```python
    labeled_image, num_features = ndimage.label(final_image)
    # Remove feature 2 which is the internal noise from light
    final_image[labeled_image == 2] = 0
```
*Example output of noise removal:*


![img_1.png](doc_imgs/img_1.png)


### Split Image at Apex
Since the profile of a pendant drop has an axis of symmetry at the apex (the lowest point of the profile),
it simplifies extraction to split the image at this point. 
```python
# find apex point(s) and split image here
final_image = split_profile(drop_profile)
```

*Final Output of Profile Extraction*


![img_2.png](doc_imgs%2Fimg_2.png)

This image(s) are saved as .png files to **dest** directory provided to the DropProfile instance.

## Feature Extraction
Feature extraction is implemented in feature_extract.py in the FeatureExtract class. Given an
ordered set of x and y coordinates from the edge profile taken in the [Profile Extraction](#Profile-Extraction) process, it will: 

* Use the final edge profile to approximate and derive the characteristic features in tabular format seen below:

| Drop Height | Capillary Radius | R-S | R-e | Apex Radius |
|-------------|------------------|-----|-----|-------------|
| -           | -                | -   | -   | -           |

Measurements of height and radii are found in terms of pixel count, and using a normalization to the apex radius, these values become dimensionless.

### Apex Radius
The apex radius is used as the normalization factor for all the other characteristic features of the profile.
This value is found by using a circle fit approximation within the drop. This function can be found in **pdt-extract/feature-extract.py* called
**find_apex_radius()**
It takes two parameters to configure the circle fit:
* **ratio_drop_length**: 1 >= float value > 0 representing number points along profile to approximate with

* **change_ro**: float value representing minimum value of change in circle radius before stopping approximation
It has been experimentally found that the parameters (0.15, .005) give the most accurate results.
#### Circle Fitting
Finding the apex radius is done by maximizing a circle size along the profile. When the radius increases to less than .005 each iteration,
the circle fit is complete and the radius is taken as apex radius.

**Circle fit approximation with a pendant drop:**

![circle-fit.gif](doc_imgs%2Fcircle-fit.gif)
### Equator Radius
Equator radius is the value of the maximum "bulge" radius. To work with smaller, less well-formed drop shapes where
the capillary radius could be larger than the equator, the equator radius is found by finding the largest x value with respect to 
the x=0 axis on the bottom 70% of the profile:

### S-Radius
S-Radius is found to be the radius at the X index: -(2 * pixel count of equator radius)
This value is inverted to take the radius of this point between the capillary and equator (from the top down).
### Capillary Radius
Capillary radius is the value of the x with respect to the (x=0) axis at the last (highest)
x coordinate:
```python
self.capillary_radius = self.x[-1]
```
### Drop Height
Since the drop profile is centered and cropped at the apex point, the drop height
is simply the value of the last y coordinate in the set of all points along the edge profile.
```
self.drop_height = self.y[0]
```
Drop height can be found by taking the value of the last y
### Normalization

Please refer to this excellent link to better understand the algorithm : "http://justin-liang.com/tutorials/canny/"

Important Points:
- I have used a relatively slow iterative approach to perform the function of Double Thresholding Hysteresis,
  a better and time-saving alternative is to use a recursive algorithm which tracks the edges.
- The value of Sigma to implement Gaussian Blur is image specific, different values can be tested to see which give the best estimate of edges.
- The ratio of the thresholds is again another variable, but the ones that I have used in the code give pretty good estimates for any particular image.
- Non Maxima Suppression with Interpolation although being computationally expensive, gives excellent estimates and is better tha NMS without interpolation
