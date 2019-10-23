import numpy as np 
import cv2
# import tensorflow as tf
from PIL import Image
import sys
sys.path.append('..')

from helper.utils import resize

def random_crop(img, pts, size):
    """
    param
    :img - origin image w=h=512
    :pts - 4 corner, shape [4x2]
    :size - new image size after crop
    
    return
    new_img, new_pts
    """
    h, w = img.shape[:2] # channel last
    new_h, new_w = size 
    ratio = h / max(new_h, new_w)
    
    min_pts = np.min(pts, axis=0)
    begin_x = min(np.random.randint(0, w - new_w), min_pts[0])
    begin_y = min(np.random.randint(0, h - new_h), min_pts[1])

    new_img = img[begin_y: begin_y + new_h, begin_x: begin_x + new_w, :]
    new_img = cv2.resize(new_img, None, fx=ratio, fy=ratio)
    pts[:, 0] -= begin_x
    pts[:, 1] -= begin_y
    # pts[pts<0] = 0

    new_img, pts = resize(new_img, pts, side=h)
    return new_img, pts

def color_variance(img, pts):
    pass

def rotate(img, pts):
    pass

def add_noise(img, pts):
    noise = np.random.randint(-20, 20, size=(img.shape))
    print(noise.shape)
    img = img.astype(np.float32)
    img += noise
    
    img[img<0] = 0 # overload number
    img[img>255] = 0 
    img = img.astype(np.uint8)

    return img, pts

# img = np.random.random((100, 100, 3))
# img = (img*255/np.max(img)).astype(np.uint8)
# pts = 1
# add_noise(img, pts)