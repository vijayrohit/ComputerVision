----------------**Please Read**----------------

Script is not self sufficient for all polygons. Few manual steps must be taken for file management. Please email if need any help with file management part of the script. Script works for 4 pointed polygons. For more than 4 points the script must be changed manually and the automatic feauture goes off.

----------------**Parameters**----------------

data.txt file has all input parameters.

Line 1: Input Target Polygon
Line 2: Threshold
Line 3: Input Source File path 
Line 4: Output File Path


Edit the data.txt file accordingly.

----------------**Input Image Specification**----------------

Make sure that required changes are made to script for file management. Code is not self sufficient for new image formats and random file sequences. 

----------------**Packages**----------------

You will need numpy, imageio, scipy, shapely, 'tracking.py' and math packages installed/used for the script to run. 

----------------**Samples**----------------

Outputs are included with scripts for basic idea of output from the script.

----------------**Functions in script**----------------

rgb_to_gray(image): converts rgb to gray_scale image
repeat_1(gray_scale, number_of_points): if points drop below threshold this repeats the 1st frame steps to current frame
correlation_matrix(ix2,iy2,ixy,image): creates a correlation matrix for each pixel in given image.
eigen_values(image): creates a 2d matrix similar to image with minimum eigenvalue of each correlation matrix of each pixel in the image.
reduce_eig_values(image): reduces the number of eigen values in the above created matrix.
get_points(image, K): provides with top k eigen points.
con_1d(image): 1D convolution for Ix and Iy is done by this function. 
con_2d(image, image1): Convolution for IxIy is done by this function.
