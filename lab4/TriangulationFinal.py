import math
import numpy as np
import cv2 
from methods import *
import sys
import time

def calibrate(pixels, worldCordinates, centre):

	v = compute_v(pixels, worldCordinates, centre[0], centre[1])

	gamma_value = gamma_abs(v[0], v[1], v[2])
	
	vnorm = normalize_v(v, gamma_value, pixels[:, 1], worldCordinates, centre[1])
	
	alpha_value = alpha(gamma_value, v[4], v[5], v[6])
	
	Tx = vnorm[7] / alpha_value
	Ty = vnorm[3]
	
	R = compute_rotation(vnorm, alpha_value)

	Tz, fx = compute_tz_fx(pixels, worldCordinates, R, Tx, centre[0])
	fy = fx/alpha_value
	
	t =[[Tx],[Ty],[Tz]]
	kM = np.array([[fx, 0, centre[0]], [0, fy, centre[1]], [0, 0, 1]])

	

	return R,kM,t

def main():


	viewFrame = 0
	totalINV = []
	totalNOR = []
	Views = ['RIGHT.jpg','LEFT.jpg']
	expLoc = 0
	file1 = open("OUTPUT.txt","a") 

	while True:
		print("Finding Geometrically Strong Points",end="")
		sys.stdout.flush()
		for i in range(5):
			time.sleep(1)
			print('.', end='', flush=True)
		print("")

		currentView = Views[viewFrame]
		img = cv2.imread(currentView)
		gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
		corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
		corners = np.int0(corners)
		corners = corners[:7]



		
		imageCoordinates = []
		Xj = []

		for corner in corners:
			x,y = corner.ravel()
			imageCoordinates.append([x,y])
			Xj.append([[x],[y],[1]])

		#print(Xj)
		imageCentre = [int(len(img[0])/2),int(len(img)/2)]


		if viewFrame == 0:
			worldCordinates = [[0,170,170], [0,0,170], [170,0,0], [170,170,0], [130,0,0], [0,0,0], [170,140,0]]
			xj = Xj[3]
			expLoc = np.asarray(worldCordinates[3])

		elif viewFrame == 1:
			worldCordinates = [[0,0,0], [0,170,170], [170,170,0], [45,170,170], [170,0,0], [170,85,0], [0,170,0]]
			xj = Xj[2]



		imageCoordinates = np.array(imageCoordinates)
		worldCordinates = np.array(worldCordinates)

		print("Calibrating Camera "+str(viewFrame),end="")
		sys.stdout.flush()
		for i in range(5):
			time.sleep(1)
			print('.', end='', flush=True)
		print("")

		R, kM, t = calibrate(imageCoordinates, worldCordinates, imageCentre)


		unitVector = np.dot(np.transpose(R),(np.matmul(np.linalg.inv(kM),xj)))


		unitVector = unitVector/np.linalg.norm(unitVector)
		

		unitVector = np.dot(unitVector,np.transpose(unitVector))
		

		TOT = np.subtract(np.identity(3),unitVector)
		

		Cj = np.dot(-1,(np.dot(np.transpose(R),t)))
		

		NOR = np.dot(TOT,Cj)
		

		totalNOR.append(NOR)
		

		totalINV.append(TOT)
		

		viewFrame += 1

		if viewFrame>1:
			break

	print("Estimating 3D Location",end="")
	sys.stdout.flush()
	for i in range(5):
		time.sleep(1)
		print('.', end='', flush=True)
	print("")
	totalI = np.linalg.inv(np.add(totalINV[0], totalINV[1]))
	totalN = np.add(totalNOR[0], totalNOR[1])
	P = np.dot(totalI,totalN)
	print("Real 3D worldCordinates",file=file1)
	print(expLoc,file=file1)
	print("Estimated 3D worldCordinates",file=file1)
	print(P,file=file1)
	print("Check OUTPUT.txt for Results")
	file1.close() 

if __name__ == '__main__':
    main()