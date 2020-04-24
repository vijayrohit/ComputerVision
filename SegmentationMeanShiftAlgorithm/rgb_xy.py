import imageio
import math
import numpy as np

def rgb_xy(input_image):
    row_count = 1
    dump = []
    for line in input_image:
        line_dump = []
        for pixel_array in line:
            line_dump.append(pixel_array.tolist())           
        dump.append(line_dump)          #Creating a dump of image array for easy manipulation
    for line in dump:
        col_count = 1
        for pixel_array in line:
            pixel_array.append(row_count)
            pixel_array.append(col_count)
            col_count+=1
        row_count+=1
    return dump

def GS_to_rgb(image):
    return np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

def rgbtoxyz(input_image):
    for line in input_image:
        for pixel_array in line:
            r = pixel_array[0]
            g = pixel_array[1]
            b = pixel_array[2]
            r = float(r/255)
            b = float(b/255)
            g = float(g/255)
            if r > 0.4045:
                r = float(((r+0.055)/1.055)**(2.4)*100)
            else:
                r = float(r/12.92*100)
            if b > 0.4045:
                b = float(((b+0.055)/1.055)**(2.4)*100)
            else:
                b = float(b/12.92*100)
            if g > 0.4045:
                g = float(((g+0.055)/1.055)**(2.4)*100)
            else:
                g = float(g/12.92*100)
            pixel_array[0] = r * 0.4124 + g * 0.3576 + b * 0.1805
            pixel_array[1] = r * 0.2126 + g * 0.7152 + b * 0.0722
            pixel_array[2] = r * 0.0193 + g * 0.1192 + b * 0.9505
    return input_image


def xyztolab(input_image):
    for line in input_image:
        for pixel_array in line:
            x = float(pixel_array[0]/100)
            y = float(pixel_array[1]/100)
            z = float(pixel_array[2]/100)
            if x > 0.008856:
                x = float(x**(1/3))
            else:
                x = float((7.787 * x) + (16.0/116))
            if y > 0.008856:
                y = float(y**(1/3))
            else:
                y = float((7.787 * y) + (16.0/116))
            if z > 0.008856:
                z = float(z**(1/3))
            else:
                z = float((7.787 * z) + (16.0/116))
            pixel_array[0] = float(((116 * y) - 16)) * 255/100
            pixel_array[1] = float(500 * (x - y)  + 128)
            pixel_array[2] = float(200 * (y - z) + 128)
    return input_image


def nearest_pixels(image, current_pixel, hs, hr, sdhs, sdhr): 
    nearest_pix = []
    for line in image:
        for pixel_array in line:
            if dist(pixel_array,current_pixel) < hr*sdhr:
                if dist_spatial(pixel_array,current_pixel) < hs*sdhs:
                    nearest_pix.append(pixel_array)
    return nearest_pix

def dist(this_pixel, other_pixel):
    return ((this_pixel[0]-other_pixel[0])**(2)+(this_pixel[1]-other_pixel[1])**(2)+(this_pixel[2]-other_pixel[2])**(2))**(0.5)


def dist_spatial(this_pixel, other_pixel):
    return ((this_pixel[3]-other_pixel[3])**(2)+(this_pixel[4]-other_pixel[4])**(2))**(0.5)


def apply_kernel(cur_pixel, adjacent_pixel, hs, hr):
    n = ((cur_pixel[0]-adjacent_pixel[0])**(2) + (cur_pixel[1]-adjacent_pixel[1])**(2) + (cur_pixel[2]-adjacent_pixel[2])**(2))/hr**(2) +((cur_pixel[3]-adjacent_pixel[3])**(2))/hs**(2) + ((cur_pixel[4]-adjacent_pixel[4])**(2))/hs**(2)
    t = -2
    return math.exp(n/t)


def remove_xy(arr_xy):
    arr=[]
    for line in arr_xy:
        line_dump=[]
        for pixel_array in line:
            pixel_dump=[]
            pixel_dump.append(pixel_array[0])
            pixel_dump.append(pixel_array[1])
            pixel_dump.append(pixel_array[2])
            line_dump.append(pixel_dump)
        arr.append(line_dump)
    return arr       

def labtoxyz(image):
    for line in image:
        for pixel_array in line:
            Y = (((pixel_array[0]*100)/255) + 16)/116
            X = ((pixel_array[1]-128)/500) + Y
            Z = Y - ((pixel_array[2]-128)/200)
            if X**3 < 0.008856:
                X = (X-(16/116))/7.787
            if Y**3 < 0.008856:
                Y = (Y-(16/116))/7.787
            if Z**3 < 0.008856:
                Z = (Z-(16/116))/7.787
            pixel_array[0] = X*100
            pixel_array[1] = Y*100
            pixel_array[2] = Z*100
            
def xyztorgb(image):
    for line in image:
        for pixel_array in line:
            X = pixel_array[0]
            Y = pixel_array[1]
            Z = pixel_array[2]
            XYZ = [X,Y,Z]
            RGB = []
            coeff = [[0.4124, 0.2126, 0.0193],[0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]]
            RGB = np.dot(XYZ,np.linalg.inv(coeff))
            R = ((((RGB[0])/100)**(1/2.4))*1.055) - 0.055
            G = ((((RGB[1])/100)**(1/2.4))*1.055) - 0.055
            B = ((((RGB[2])/100)**(1/2.4))*1.055) - 0.055
            if R < 0.4045:
                R = (R/100) * 12.92
            if G < 0.4045:
                G = (G/100) * 12.92
            if B < 0.4045:
                B = (B/100) * 12.92
            pixel_array[0] = (R*255)
            pixel_array[1] = (G*255)
            pixel_array[2] = (B*255)            
            
    
    
with open('data.txt', 'r') as myfile:
    parameters = myfile.readlines()

filename = ""
for i in list(parameters[0]):
    if i != "\n":   
        filename = filename + i
m = int(parameters[1])
M = int(parameters[2])
hr = int(parameters[3])
hs = int(parameters[4])
sdhr = int(parameters[5])
sdhs = int(parameters[6])
result_img = parameters[7]

image = imageio.imread(filename)

if len(image[0][0])==1:
    image = GS_to_rgb(image)

pixel_location = rgb_xy(image)
pixel_location = rgbtoxyz(pixel_location)
pixel_location = xyztolab(pixel_location)

result = []
copy_image = []
for i in range(0,m):
    for row in pixel_location:
        copy_image.append(row)
    for row in pixel_location:
        for pixel in row:
            adj_pixels = nearest_pixels(copy_image,pixel,hs, hr, sdhs, sdhr)
            if len(adj_pixels) > M:
                Neighborhood_weight = 0
                L = []
                a = []
                b = []
                for adj in adj_pixels:
                    L.append(adj[0]*apply_kernel(pixel,adj, hs, hr))
                    a.append(adj[1]*apply_kernel(pixel,adj, hs, hr))
                    b.append(adj[2]*apply_kernel(pixel,adj, hs, hr))
                    Neighborhood_weight+= apply_kernel(pixel,adj, hs, hr)
                pixel[0] = sum(L)/Neighborhood_weight
                pixel[1] = sum(a)/Neighborhood_weight
                pixel[2] = sum(b)/Neighborhood_weight

result = remove_xy(pixel_location)
                
labtoxyz(result)
xyztorgb(result)
imageio.imwrite(result_img, np.asarray(result)) 


                