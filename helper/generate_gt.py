import numpy as np 
import cv2 

def gauss(x, y, sigma=1):
    pass

def gerarate_gt():
    pass

def generate_offset(size, ratio):

    height, width = size

    offset_map = np.zeros((height, width, 2)) # 2 offset for x and y

    for i in range(height):
        for j in range(width):
            offset_map[i, j, :] = (i/ratio - np.floor(i/ratio),
                                j/ratio - np.floor(j/ratio))
    
    
    return offset_map


