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
* Upload pendant drop images as PNG to the **path** subdirectory
This tool currently only works with .png images. It also assumes that the images are already cropped, removing the capillary.
See **Pendant Drops** subdirectory for valid sample images.

This will output the extracted profiles to the subdirectory Drop Profiles or **dest** argument for each image

To use this package:
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

### Double Hysterisis Thresholding
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

*Example output of canny sequence:*


![img.png](doc_imgs/img.png)

### Reflective Noise Removal
The smaller internal edge is detected from the reflected light of the pendant drop when
the image is taken. Since this edge is connected and assumed to be always smaller than the edge profile, the smaller edge is isolated and filterd out:
```python
    labeled_image, num_features = ndimage.label(final_image)
    # Remove feature 2 which is the internal noise from light
    final_image[labeled_image == 2] = 0
```
*Example output of noise removal:*


![img_1.png](doc_imgs/img_1.png)


### Split Image at Apex
Since the profile of a pendant drop has an axis of symmetry at the apex (lowest point of the profile),
it simplifies extraction to split the image at this point. 
```python
# find apex point(s) and split image here
final_image = split_profile(drop_profile)
```

*Final Ouput of Profile Extraction*


![img_2.png](doc_imgs%2Fimg_2.png)

This image(s) are saved as .png files to **dest** directory provided to the DropProfile instance.

## Feature Extraction

This tool uses the final edge profile to approximate and derive the characteristic features in tabular format seen below:

| Drop Height | Capillary Radius | R-S | R-e |
|-------------|------------------|-----|-----|
| -           | -                | -   | -   | 

### Circle Fit

Please refer to this excellent link to better understand the algorithm : "http://justin-liang.com/tutorials/canny/"

Important Points:
- I have used a relatively slow iterative approach to perform the function of Double Thresholding Hysterisis,
  a better and time-saving alternative is to use a recursive algorithm which tracks the edges.
- The value of Sigma to implement Gaussian Blur is image specific, different values can be tested to see which give the best estimate of edges.
- The ratio of the thresholds is again another variable, but the ones that I have used in the code give pretty good estimates for any particular image.
- Non Maxima Suppression with Interpolation although being computationally expensive, gives excellent estimates and is better tha NMS without interpolation
