import numpy as np
def compute_v(points2D, points3D, ox, oy):
    assert points2D.shape[0] == points3D.shape[0]
    
    x = points2D[:, 0] - ox
    y = points2D[:, 1] - oy
    X = points3D[:, 0]    
    Y = points3D[:, 1]        
    Z = points3D[:, 2]
    
    xX = x*X
    xY = x*Y 
    xZ = x*Z
    yX = -y*X
    yY = -y*Y
    yZ = -y*Z
    
    A = np.column_stack((xX, xY, xZ, x, yX, yY, yZ, -y))
    assert A.shape[0] == points2D.shape[0]
    U, D, V = np.linalg.svd(A)

    return V.T[:, -1]

def gamma_abs(v1, v2, v3):
    return np.sqrt(v1*v1 + v2*v2 + v3*v3)

def normalize_v(v, gamma, y, points3D, oy):
    vnorm = v / gamma
    Ty = vnorm[3]
    sig = (y[0] - oy) * (vnorm[0]*points3D[0][0] + vnorm[1]*points3D[0][1] + vnorm[2]*points3D[0][2] + vnorm[3])
    if sig > 0:
        return -vnorm
    return vnorm

def alpha(gamma, v5, v6, v7):
    return np.sqrt(v5*v5 + v6*v6 + v7*v7) / gamma

def compute_rotation(vnorm, alpha_value, ortho=False):
    r1 = vnorm[4:7] / alpha_value
    r2 = vnorm[0:3]
    r3 = np.cross(r1, r2)
    R = np.array([r1, r2, r3])
    return R

def compute_tz_fx(points2D, points3D, R, Tx, ox):
    x = points2D[:, 0] - ox
    second_col = np.column_stack((points3D, Tx * np.ones(points3D.shape[0]))).dot(np.append(R[0, :], 1))
    #print(second_col)
    A = np.column_stack((-x, second_col))

    b = x * points3D.dot(R[2, :])
    x = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
    return x[0], x[1]

