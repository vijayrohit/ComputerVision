import numpy as np
from scipy.spatial import distance

def matA(pixelLocation, worldCordinates, opticCenter):
	A = []
	for i in range(len(pixelLocation)):
		xs,ys = (pixelLocation[i][0]-opticCenter[0]), (pixelLocation[i][1]-opticCenter[1])
		Xw,Yw,Zw = worldCordinates[i][0], worldCordinates[i][1], worldCordinates[i][2]
		A.append([Xw*xs, Yw*xs, Zw*xs, xs, Xw*ys, Yw*ys, -ys])
	S, V, D = np.linalg.svd(A)
	A = D.T[:,-1]
	#print(A)
	return A


def matb(pixelLocation, worldCordinates, opticCenter):
	b = []
	for i in range(len(pixelLocation)):
		xs,ys = (pixelLocation[i][0]-opticCenter[0]), (pixelLocation[i][1]-opticCenter[1])
		Xw,Yw,Zw = worldCordinates[i][0], worldCordinates[i][1], worldCordinates[i][2]
		b.append([Zw*ys])
	return b


def makeOthrogonal(v1, v2):
	
	while True:
		if np.dot(v1,v2)!=0:
			delta = np.dot(v1,v2)/np.dot(v1,v1)
			v1New = np.dot(delta,v1)
			v2 = np.subtract(v2,v1New)
		else:
			break

	return v2



def compute_tz_fx(points2D, points3D, R, Tx, ox):
    x = points2D[:, 0] - ox
    second_col = np.column_stack((points3D, Tx * np.ones(points3D.shape[0]))).dot(np.append(R[0, :], 1))
    print(second_col)
    A = np.column_stack((-x, second_col))
    
    b = x * points3D.dot(R[2, :])
    x = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
    return x[0], x[1]



def findTzF(R, Ty, pixelLocation, worldCordinates, opticCenter):
	maxDistance = 0
	Yi = 0

	for i in range(len(pixelLocation)):

		xs,ys = (pixelLocation[i][0]-opticCenter[0]), (pixelLocation[i][1]-opticCenter[1])
		thisDistance = np.sqrt(xs**2 + ys**2)

		if maxDistance < thisDistance:
			maxDistance = thisDistance
			Yi = worldCordinates[i][1]
	#print(Yi)
	rightCoef = []
	leftCoef = []
	for i in [2,4]:
		yi = R[1][0]*worldCordinates[i][0] + R[1][1]*worldCordinates[i][1]+Ty
		wi = R[2][0]*worldCordinates[i][0] + R[2][1]*worldCordinates[i][1]
		leftCoef.append([float(yi), -Yi])
		rightCoef.append([wi*Yi])
	invl = np.linalg.inv(leftCoef)
	#print(leftCoef)
	#print(rightCoef)
	return np.dot(invl, rightCoef)



def calibrateCamera(pixelLocation, worldCordinates, opticCenter):

	A = matA(pixelLocation, worldCordinates, opticCenter)
	b = matb(pixelLocation, worldCordinates, opticCenter)
	#p = np.dot(np.linalg.inv(A),b)


	c1 = np.sqrt((A[0]*A[0])+(A[1]*A[1])+(A[2]*A[2]))
	c2 = np.sqrt((A[3]*A[3])+(A[4]*A[4])+(A[5]*A[5]))


	aspectRatio = c2/c1
	Rx = [float(A[0]/c1), float(A[1]/c1), float(A[2]/c1)]
	Ry = [float(A[3]/c2), float(A[4]/c2), float(A[5]/c2)]
	#print(Rx,Ry)
	Rz = np.cross(Rx,Ry)
	R = [Rx, Ry, Rz]
	R = np.asarray(R)
	#print(R)
	Tx = A[6]/c1
	Ty = 1/c1
	#print(Ty)

	Tz,f = compute_tz_fx(np.array(pixelLocation),np.asarray(worldCordinates), R, Tx, opticCenter[0] )


	return float(aspectRatio), f, np.asarray([[float(Tx)],[float(Ty)],[Tz]]), R




	


