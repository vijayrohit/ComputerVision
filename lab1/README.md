**Mean Shift Algorithm**

----------------**Parameters**----------------

data.txt file has all input parameters.

Line 1: Input File Path
Line 2: Number of iterations (m)
Line 3: Cluster Threshold (M)
Line 4: Color Bandwidth (hr)
Line 5: Spatial Bandwidth (hs)
Line 6: SD of Color Bandwidth (sdhr)
Line 7: SD of Spatial Bandwidth (sdhs)
Line 8: Output File Path

Edit the data.txt file accordingly.

----------------**Input Image Size**----------------

60x60 image will take 10 mins for excution. 128x128 image with 5 iterations will take 1 hour for execution.

----------------**Packages**----------------

you will need numpy, imageio and math packages installed for the script to run. 

----------------**Samples**----------------

sample_input and sample_outputs are included with scripts for basic idea of output from the script.

----------------**Functions in script**----------------

rgb_xy(image): adds (x,y) to the image
rgbtoxyz(image): converts [r,g,b] to [x,y,z] space
xyztolab(image): converts [x,y,z] to [l,a,b]
nearest_pixels(image, current_pixel, hs, hr, sdhs, sdhr): provides a list of pixels which are closest to the current pixel in spatial range and color range. 
dist(this_pixel, other_pixel): computes the euclidean distance between 2 pixels in color space.
dist_spatial(this_pixel, other_pixel): computes the euclidean distance between 2 pixels in spatial space.
apply_kernel(cur_pixel, adjacent_pixel, hs, hr): computes the weight of a neighboring pixel of a current pixel. For this assignment, gaussian kernel is used. 
remove_xy(image): removes the (x,y) component from the pixels of the image
labtoxyz(image): converts [l,a,b] to [x,y,z] color space
xyztorgb(image): converts [x,y,z] to [r,g,b] color space
GS_to_rgb(image): converts a grayscale pixel to 3d rgb pixel.
