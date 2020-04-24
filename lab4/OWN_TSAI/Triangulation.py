import math
import numpy as np
import cv2
from calibrate import calibrateCamera 
from methods import *


img = cv2.imread('RIGHT.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

corners = corners[:7]
#print(corners)
pixels1 = []
xj1 = []
for corner in corners:
	x,y = corner.ravel()
	pixels1.append([x,y])
	xj1.append([[x],[y],[1]])
centre1 = [int(len(img[0])/2),int(len(img)/2)]
worldCordinates1 = [[0,170,170], [0,0,170], [170,0,0], [170,170,0], [130,0,0], [0,0,0], [170,140,0]]

a1, f1, t1, R1 = calibrateCamera(pixels1, worldCordinates1, centre1)


kM = [[-f1,0,centre1[0]],[0,-(a1*f1),centre1[1]],[0,0,1]]

v1 = np.dot(np.transpose(R1),(np.matmul(np.linalg.inv(kM),xj1[3])))
v1 = v1/np.linalg.norm(v1)
vf1 = np.dot(v1,np.transpose(v1))
#print(vf1)
tot1 = np.subtract(np.identity(3),vf1)
cj1 = np.dot(-1,(np.dot(np.transpose(R1),t1)))
s21 = np.dot(tot1,cj1)

#print(a1)
#print(f1)
#print(t1)
#print(R1)

img2 = cv2.imread('LEFT.jpg')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray2,25,0.01,10)
corners = np.int0(corners)

corners = corners[:7]
#print(corners)
pixels2 = []
xj2 = []
for corner in corners:
	x,y = corner.ravel()
	pixels2.append([x,y])
	xj2.append([[x],[y],[1]])
centre2 = [int(len(img2[0])/2),int(len(img2)/2)]
worldCordinates2 = [[0,0,0], [0,170,170], [170,170,0], [45,170,170], [170,0,0], [170,85,0], [0,170,0]]
			


a2, f2, t2, R2 = calibrateCamera(pixels2, worldCordinates2, centre2)
##print(centre1)
#print(centre2)
kM2 = [[-f2,0,centre2[0]],[0,-(a2*f2),centre2[1]],[0,0,1]]
v2 = np.dot(np.transpose(R2),(np.matmul(np.linalg.inv(kM2),xj2[2])))
v2 = v2/np.linalg.norm(v2)
vf2 = np.dot(v2,np.transpose(v2))
#print(vf2)
tot2 = np.subtract(np.identity(3),vf2)
cj2 = np.dot(-1,(np.dot(np.transpose(R2),t2)))
s22 = np.dot(tot2,cj2)
toti = np.linalg.inv(np.add(tot1,tot2))
s2 = np.add(s21,s22)
P = np.dot(toti,s2)
print(P)