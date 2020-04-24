#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
import sys
import imageio
import math
from scipy.ndimage import gaussian_filter1d
import os
from tracking import rgb_to_gray, con_2d, con_1d, correlation_matrix, eigen_values, reduce_eig_values, get_points
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#from Left_overs import createJacobian, createHesian, createBArray, createAffine

# Creating Delta Array using Source Corners and Target Corners


def createDelta(srcCorners, tgtCorners):
    deltaArray = []

    for i in range(0, len(srcCorners)):
        x = -(tgtCorners[i][0] - srcCorners[i][0])
        y = -(tgtCorners[i][1] - srcCorners[i][1])
        deltaArray.append([[x], [y]])

    return deltaArray


# Creating Source corners array


def srcEdges(rows, columns):
    return [[0, 0], [0, columns], [rows, 0], [rows, columns]]


# Reverse Assigning the pixels of source to target


def Homography(tgtCorners, srcCorners, affArray, changeElements, image,
               srcImage, final, poly):

    residual = 0
    newLocation = []
    for i in range(0, len(tgtCorners)):

        cordMat = [[srcCorners[i][0]], [srcCorners[i][1]], [1]]

        h20 = changeElements[6][0]
        h21 = changeElements[7][0]
        D = h20 * srcCorners[i][0] + h21 * srcCorners[i][1] + 1

        tgtCod = np.floor(np.matmul(affArray, cordMat))

        # print(tgtLoc)

        x = int(tgtCod[0][0] / D)
        y = int(tgtCod[1][0] / D)
        residual += np.linalg.norm([[tgtCorners[i][0] - x],
                                    [tgtCorners[i][1] - y]])
        newLocation.append([x, y])

    if final == True:
        #print(srcCorners)
        polygon = Polygon(poly)
        for i in range(0, len(image)):
            for j in range(0, len(image[0])):
                point = Point(i, j)
                if polygon.contains(point):
                    cordMat = [[i], [j], [1]]

                    tgtLoc = np.floor(np.matmul(affArray, cordMat))

                    h20 = changeElements[6][0]
                    h21 = changeElements[7][0]
                    D = h20 * i + h21 * j + 1

                    x = int(round(tgtLoc[0][0] / D))
                    y = int(round(tgtLoc[1][0] / D))

                    if x < len(srcImage) and y < len(
                            srcImage[0]) and x >= 0 and y >= 0:
                        image[i][j][0] = srcImage[x][y][0]
                        image[i][j][1] = srcImage[x][y][1]
                        image[i][j][2] = srcImage[x][y][2]

    return (residual, newLocation)


# Finding Jacobian


def homoJacobian(points, tgtp, changeElements):
    jacArray = []

    h20 = changeElements[6][0]
    h21 = changeElements[7][0]

    # print(points)

    for i in range(0, len(tgtp)):
        D = h20 * points[i][0] + h21 * points[i][1] + 1

        # xc = ((1+h00)*points[i][0] + h01*points[i][1] + h02)/D
        # yc = (h10*points[i][0] + (h11+1)*points[i][1] + h12)/D

        jacArray.append(
            int(1 / D) * np.array([[
                points[i][0],
                points[i][1],
                1,
                0,
                0,
                0,
                -(points[i][0] * tgtp[i][0]),
                -(points[i][1] * tgtp[i][0]),
            ], [
                0,
                0,
                0,
                points[i][0],
                points[i][1],
                1,
                -(points[i][0] * tgtp[i][1]),
                -(points[i][1] * tgtp[i][1]),
            ]]))

    # print(jacArray)

    return jacArray


# Creating Hesian Matrix


def homoHesian(jacArray):
    hesArray = np.zeros((8, 8))
    for i in range(0, len(jacArray)):
        jacArrayTrans = np.transpose(jacArray[i])

        # print(jacArrayTrans)
        # print(np.dot(jacArrayTrans,jacArray[i]))

        hesArray = np.add(hesArray, np.dot(jacArrayTrans, jacArray[i]))
    return hesArray.astype(int)


# Creating b Array for Parameter Matrix calculation


def homoBArray(jacArray, deltaArray):
    BArray = np.zeros((8, 1))

    for i in range(0, len(jacArray)):
        jacArrayTrans = np.transpose(jacArray[i])
        BArray = np.add(BArray, np.matmul(jacArrayTrans, deltaArray[i]))

    return BArray.astype(int)


# Find the parameter matrix


def changeEl(hesArray, BArray, esp):
    hessianInv = np.linalg.inv(
        hesArray + np.dot(esp, np.diag(np.diag(hesArray))))

    # print(hessianInv.astype(float))

    return np.matmul(hessianInv, BArray).astype(float)


# Creating H Array


def createHomoArray(changeElements):
    homoArray = []

    h00 = changeElements[0][0]
    h01 = changeElements[1][0]
    h02 = changeElements[2][0]
    h10 = changeElements[3][0]
    h11 = changeElements[4][0]
    h12 = changeElements[5][0]
    h20 = changeElements[6][0]
    h21 = changeElements[7][0]

    homoArray = [[h00 + 1, h01, h02], [h10, h11 + 1, h12], [h20, h21, 1]]

    return homoArray


def main():
    with open('data.txt', 'r') as myfile:
        parameters = myfile.readlines()

    filename = ""
    for i in list(parameters[0]):
        if i != "\n":
            filename = filename + i

    thresholdResidual = int(parameters[1])

    source = ""
    for i in list(parameters[2]):
        if i != "\n":
            source = source + i

    output = ""
    for i in list(parameters[3]):
        if i != "\n":
            output = output + i

    esp = float(parameters[4])
    image = imageio.imread(filename)

    # print(len(image),len(image[0]))

    srcImage = imageio.imread(source)
    srcCorners = []
    srcCorners = srcEdges(len(srcImage), len(srcImage[0]))
    if filename == 'target/pepsiTruck.png':
        tgtCorners = [[58, 1046], [283, 1569], [557, 1055], [519, 1578]]
        poly = [tgtCorners[0], tgtCorners[2], tgtCorners[3], tgtCorners[1]]

    elif filename == 'target/hallway.png':
        tgtCorners = [[492, 0], [54, 1100], [1225, 0], [1975, 1100]]
        poly = [tgtCorners[0], tgtCorners[2], tgtCorners[3], tgtCorners[1]]

    elif filename == 'target/hallway_30.png':
        tgtCorners = [[150, 0], [16, 330], [370, 0], [590, 330]]
        poly = [tgtCorners[0], tgtCorners[2], tgtCorners[3], tgtCorners[1]]

    elif filename == 'target/stopSign.jpg':
        tgtCorners = [[38, 82], [30, 139], [70, 182], [142, 181], [195, 136],
                      [194, 76], [149, 38], [87, 40]]
        poly = [[38, 82], [30, 139], [70, 182], [142, 181], [195, 136],
                [194, 76], [149, 38], [87, 40]]
        if source == 'source/cokeBig.png':
            srcCorners = [[0, 427], [0, 854], [240, 1279], [480, 1279],
                          [719, 854], [719, 427], [480, 0], [240, 0]]
        elif source == 'source/coke.png':
            srcCorners = [
                [0, 100],
                [0, 200],
                [56, 299],
                [112, 299],
                [168, 200],
                [168, 100],
                [112, 0],
                [56, 0],
            ]
        elif source == 'source/nerdy.png':
            srcCorners = [[0, 200], [0, 400], [117, 599], [234, 599],
                          [349, 400], [349, 200], [234, 0], [117, 0]]
        if source == 'source/poca.jpg':
            srcCorners = [[0, 427], [0, 854], [240, 1279], [480, 1279],
                          [719, 854], [719, 427], [480, 0], [240, 0]]

    else:
        gray_scale = rgb_to_gray(image)

        x_gradient = gaussian_filter1d(
            gaussian_filter1d(gray_scale, sigma=1, axis=0),
            sigma=1,
            order=1,
            axis=1)

        y_gradient = gaussian_filter1d(
            gaussian_filter1d(gray_scale, sigma=1, axis=1),
            sigma=1,
            order=1,
            axis=0)

        xy_gradient = con_2d(x_gradient, y_gradient)

        xx_gradient = con_1d(x_gradient)

        yy_gradient = con_1d(y_gradient)

        cor_mat = correlation_matrix(xx_gradient, yy_gradient, xy_gradient,
                                     gray_scale)

        eig_val_mat = eigen_values(cor_mat)

        reduce_eig_values(eig_val_mat)

        tgtCorners = get_points(eig_val_mat, 4)

        poly = [tgtCorners[0], tgtCorners[2], tgtCorners[3], tgtCorners[1]]

    print("Target corners and Source corners : Created!")
    print(tgtCorners)

    print("Creating Homography matrix",end="")
    sys.stdout.flush()
    for i in range(5):
        time.sleep(1)
        print('.', end='', flush=True)
    print("")
    

    p = [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]

    #jacArray = createJacobian(srcCorners)
    #deltaArray = createDelta(tgtCorners, srcCorners)
    #hesArray = createHesian(jacArray)
    #BArray = createBArray(jacArray, deltaArray)
    #p,AffineArray = createAffine(hesArray, BArray)

    # residual, newLocation = Homography(tgtCorners, srcCorners, AffineArray)
    H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    (residual, newLocation) = Homography(srcCorners, tgtCorners, H, p, image,
                                         srcImage, False, tgtCorners)
    #print(srcCorners)
    # print(H)
    print(residual, newLocation)
    while residual > thresholdResidual:
        jacArray = homoJacobian(newLocation, srcCorners, p)

        #print(jacArray)

        deltaArray = createDelta(srcCorners, newLocation)

        #print(deltaArray)

        hesArray = homoHesian(jacArray)

        #print(hesArray)

        BArray = homoBArray(jacArray, deltaArray)

        # print(BArray)

        dp = changeEl(hesArray, BArray, esp)

        # print(dp)

        p = np.add(p, dp)

        # print(p)

        H = createHomoArray(p)

        #print(H)

        (residual, newLocation) = Homography(
            srcCorners, newLocation, H, p, image, srcImage, False, tgtCorners)

        print(residual, newLocation)
    # print(H)
    # mapPixel(image, srcImage, H,p)
    (residual, newLocation) = Homography(srcCorners, newLocation, H, p, image,
                                         srcImage, True, poly)
    


    print("Output Generating",end='')
    sys.stdout.flush()
    for i in range(5):
        time.sleep(1)
        print('.', end='', flush=True)
    print("")


    # Output
    imageio.imwrite(output, image)


if __name__ == '__main__':
    main()
