import numpy as np 
import cv2 

def gaussian2D(x, y, sigma=1):
    heatmap = np.zeros((x, y))
    center = [x//2, y//2]
    for i in range(x):
        for j in range(y):
            heatmap[i, j] = 1 / (2*np.pi*sigma**2) * np.exp(-((i-center[0])**2+(j-center[1])**2)/2)
    return heatmap


def gerarate_gt():
    pass

def generateOffset(size, ratio):

    height, width = size

    new_h, new_w = height//ratio, width//ratio

    offset_map = np.zeros((new_h, new_w, 2), dtype='float32')

    for i in range(new_h):
        for j in range(new_w):
            offset_map[i, j, :] = ()
    
    
    return offset_map


