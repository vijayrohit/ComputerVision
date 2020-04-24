----------------**Please Read**----------------

Script is not self sufficient for all scenarios. Few manual steps must be taken for file management. Please email if need any help with file management part of the script.

----------------**Input Image Specification**----------------

Make sure you use the same input images. Change of images need a change in 3D coordinates accordingly. 

----------------**Packages**----------------

You will need numpy, cv2, scipy, sys, time, 'method.py' and math packages installed/used for the script to run. 

----------------**Samples**----------------

Outputs are included with scripts for basic idea of output from the script. (OUTPUT.txt)

----------------**Functions in script**----------------

compute_v(points2D, points3D, ox, oy): Computes the initial parameter matrix
gamma_abs(v1, v2, v3): Computes the constant (C1)
normalize_v(v, gamma, y, points3D, oy): Normalizes the parameter matrix constructed initially
alpha(gamma, v5, v6, v7): Computes the constant (C2)
compute_rotation(vnorm, alpha_value, ortho=False): Computes the Rotation Matrix
compute_tz_fx(points2D, points3D, R, Tx, ox): Computes initial estimates of Tz and Fx
calibrate(pixels, worldCordinates, centre): Calls all the above functions and provides us with R, t and K matrices



Corner Detection: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html

TSAI Calibration: https://github.com/gpmarques/tsai