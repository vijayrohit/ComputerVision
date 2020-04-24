import numpy as np
import glob
import imageio
import math
from scipy.ndimage import gaussian_filter1d
import os

#converts rgb to gray_scale image
def rgb_to_gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.144])


# if points drop below threshold this repeats the 1st frame steps to current frame
def repeat_1(gray_scale, number_of_points):
    x_gradient = gaussian_filter1d(gaussian_filter1d(gray_scale,
                                       sigma=1, axis=0), sigma=1, order=1,
                                       axis=1)
    y_gradient = gaussian_filter1d(gaussian_filter1d(gray_scale,
                                   sigma=1, axis=1), sigma=1, order=1,
                                   axis=0)
    xy_gradient = con_2d(x_gradient, y_gradient)
    xx_gradient = con_1d(x_gradient)
    yy_gradient = con_1d(y_gradient)
    cor_mat = correlation_matrix(xx_gradient, yy_gradient, xy_gradient,
                                 gray_scale)
    eig_val_mat = eigen_values(cor_mat)
    reduce_eig_values(eig_val_mat)
    points = get_points(eig_val_mat, number_of_points)
    return points


#creates a correlation matrix for each pixel in given image.
def correlation_matrix(ix2,iy2,ixy,image):
    (row, col) = (len(ix2), len(ix2[0]))
    return_image = []
    for i in range(0, row):

        line = []
        for j in range(0, col):
            m = [[ix2[i][j], ixy[i][j]], [ixy[i][j], iy2[i][j]]]
            line.append(m)
        return_image.append(line)
    return np.asarray(return_image)


#creates a 2d matrix similar to image with minimum eigenvalue of each correlation matrix of each pixel in the image.
def eigen_values(image):
    return_image = []
    for i in image:
        line = []
        for j in i:
            eigvals = np.linalg.eigvals(j)
            mineig = min(eigvals)
            line.append(mineig)
        return_image.append(line)
    return return_image


#reduces the number of eigen values in the above created matrix.
def reduce_eig_values(image):
    (row, col) = (len(image), len(image[0]))
    vals = []
    for i in range(2, row - 2):
        for j in range(2, col - 2):
            maxeig = 0
            for u in range(-2, 3):
                for v in range(-2, 3):
                    if image[i + u][j + v] > maxeig:
                        maxeig = image[i + u][j + v]
            for u in range(-2, 3):
                for v in range(-2, 3):
                    if image[i + u][j + v] < maxeig:
                        image[i + u][j + v] = 0


#provides with top k eigen points.                      
def get_points(image, K):
    (row, col) = (len(image), len(image[0]))
    points = []
    for i in range(0, row):
        for j in range(0, col):
            if image[i][j] > 0:
                li = [image[i][j], i, j]
                points.append(li)
    points = sorted(points)
    points = points[-K:]
    only_co = []
    for i in points:
        only_co.append([i[1],i[2]])
    return only_co


#1D convolution for Ix and Iy is done by this function. 
def con_1d(image):
    (row, col) = (len(image), len(image[0]))
    return_image = np.zeros((row, col))
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            gauss_weight = 0
            for u in range(-1, 2):
                for v in range(-1, 2):
                    gauss_weight += image[i + u][j + v] ** 2
            return_image[i, j] = int(gauss_weight)
    return return_image


#Convolution for IxIy is done by this function.
def con_2d(image, image1):
    (row, col) = (len(image), len(image[0]))
    return_image = np.zeros((row, col))
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            gauss_weight = 0
            for u in range(-1, 2):
                for v in range(-1, 2):
                    gauss_weight += image[i + u][j + v] * image1[i
                            + u][j + v]
            return_image[i, j] = int(gauss_weight)
    return return_image


#Given tracking points this will print the point with [255,0,0](red) in the input frame.
def print_points(image,points):
    for point in points:
        x = point[1]
        y = point[2]
        image[x][y] = [255, 0, 0]


