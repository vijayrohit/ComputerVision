import imageio
import numpy as np
import glob 
import math
from scipy.ndimage import gaussian_filter1d

#Converts RGB image to GRAY_SCALE image
def convert_to_gray_scale(frame):
    gray_image = []
    for row in frame:
        gray_row = []
        for pixel in row:
            x = (pixel[0]*0.299) + (pixel[1]*0.587) + (pixel[2]*0.144)
            gray_row.append(x)
        gray_image.append(gray_row)
    return gray_image
  
    
#Convolution of Ix,Iy to Ixx, Iyy, IxIy
def gaus_window(frame1, frame2):
    temp_frame = []
    for x in range(2,len(frame1)-2):
        row = []
        for y in range(2,len(frame1[0])-2):
            total = 0
            for u in range(-2,3):
                for v in range(-2,3):
                    total+= frame1[x+u][y+v]*frame2[x+u][y+v]
            row.append(total)
        temp_frame.append(row)
    return temp_frame


#Creation of AC matrix
def auto_correlation(frame, Ixx, Iyy, IxIy):
    row = len(Ixx)
    col = len(Ixx[0])
    temp_matrix = []
    for i in range(0,row):
        r = []
        for j in range(0,col):
            A = [ [Ixx[i][j], IxIy[i][j]], [IxIy[i][j], Iyy[i][j]] ]
            r.append(A)
        temp_matrix.append(r)
    return temp_matrix


#Creation of eigen values for the auto-correlation matrix
def eigens(A_matrix):
    eigen_list = []
    for row in A_matrix:
        r = []
        for column in row:
            eigen_value = np.linalg.eigvals(column)
            r.append(min(eigen_value))
        eigen_list.append(r)
    for x in range(2,len(eigen_list)-2):
        for y in range(2,len(eigen_list[0])-2):
            local_max = 0
            for u in range(-2,3):
                for v in range(-2,3):
                    if eigen_list[x+u][y+v]>local_max:
                        local_max = eigen_list[x+u][y+v]
            for u in range(-2,3):
                for v in range(-2,3):
                    if eigen_list[x+u][y+v]<local_max:
                        eigen_list[x+u][y+v] = 0
    return eigen_list
   
    
#Selection of best points to track
def best_points_to_track(eigen_list, top_k):
    points = []
    for i in range(0,len(eigen_values)):
        for j in range(0,len(eigen_values[0])):
            if eigen_values[i][j]>0:
                points.append([eigen_values[i][j],i,j])
    points = sorted(points)
    return points[-top_k:]


#Reading the frames 
with open('variables.txt', 'r') as myfile:
    data = myfile.readlines()

f = data[0].replace("\n","")
o = data[1].replace("\n","") 
req_points = int(data[2])
floor = int(data[3])
if f=="moon_frames/":
    fi = glob.glob("moon_frames/*.png")
    frames = []
    for i in range(0,len(fi)):
        frames.append(f+str(i)+".png")
else:
    frames=glob.glob(f+"/*.jpg")
position_of_good_points = []
file_no = 0
for frame in frames:
    current_frame = imageio.imread(frame)
    gray_image = convert_to_gray_scale(current_frame)
    if frames.index(frame) == 0:
        Ix = gaussian_filter1d((gaussian_filter1d(gray_image, sigma=1, axis=0)), sigma=1, order=1, axis=1)
        Ixx = gaus_window(Ix,Ix)
        Iy = gaussian_filter1d((gaussian_filter1d(gray_image, sigma=1, axis=1)), sigma=1, order=1, axis=0)
        Iyy = gaus_window(Iy,Iy)
        IxIy = gaus_window(Ix,Iy)
        auto_correlation_matrix = auto_correlation(gray_image, Ixx, Iyy, IxIy)
        eigen_values = eigens(auto_correlation_matrix)
        good_points = best_points_to_track(eigen_values, req_points)
        #Creates points
        for i in good_points:
            x = i[1]
            y = i[2]
            position_of_good_points.append([gray_image[x][y], x, y])
            current_frame[x][y] = [255,0,0]
        imageio.imwrite(o+str(file_no)+".png",current_frame)
        file_no+=1
        print(file_no)
    else:
        for good_point in position_of_good_points:
            summed_square_difference_list = []
            for x in range(-8,9):
                for y in range(-8,9):
                    summed_square_difference = 0
                    x_coord = x + good_point[1]
                    y_coord = y + good_point[2]
                    if x_coord < len(gray_image)-1 and y_coord < len(gray_image[0])-1:
                        for u in range(-1,2):
                            for v in range(-1,2):                               
                                summed_square_difference+= ((gray_image[x_coord + u][y_coord + v]) - good_point[0])**(2)
                        summed_square_difference_list.append([summed_square_difference, gray_image[x_coord][y_coord], x_coord, y_coord])
            summed_square_difference_list = sorted(summed_square_difference_list)
            position_of_good_points[position_of_good_points.index(good_point)][0]=summed_square_difference_list[0][1]
            position_of_good_points[position_of_good_points.index(good_point)][1]=summed_square_difference_list[0][2]
            position_of_good_points[position_of_good_points.index(good_point)][2]=summed_square_difference_list[0][3]
            
        #Removes overlapping points
        no_duplicates = set(tuple(x) for x in position_of_good_points)
        position_of_good_points = [ list(x) for x in no_duplicates ]
        
        #Adds new points
        if len(position_of_good_points) < floor:
            Ix = gaussian_filter1d((gaussian_filter1d(gray_image, sigma=1, axis=0)), sigma=1, order=1, axis=1)
            Ixx = gaus_window(Ix,Ix)
            Iy = gaussian_filter1d((gaussian_filter1d(gray_image, sigma=1, axis=1)), sigma=1, order=1, axis=0)
            Iyy = gaus_window(Iy,Iy)
            IxIy = gaus_window(Ix,Iy)
            auto_correlation_matrix = auto_correlation(gray_image, Ixx, Iyy, IxIy)
            eigen_values = eigens(auto_correlation_matrix)
            good_points = best_points_to_track(eigen_values, req_points - len(position_of_good_points))
            for x in good_points:
                position_of_good_points.append([gray_image[x[1]][x[2]], x[1], x[2]])
            #Creates points
            for i in position_of_good_points:
                x = i[1]
                y = i[2]
                current_frame[x][y] = [255,0,0]
        else:                                       
            for i in position_of_good_points:
                x = i[1]
                y = i[2]
                current_frame[x][y] = [255,0,0]
        file_no+=1
        print(file_no)
        imageio.imwrite(o+str(file_no)+".png",current_frame)
    
        
    
    

    
    
    