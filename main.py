from detector import ObjectDetection
import cv2 
from PIL import Image
import numpy as np
from helper.utils import resize_image

def load_image(im_path):
    image = Image.open(im_path).convert('RGB')
    image = np.array(image)
    return image
    
objDet = ObjectDetection()
im = load_image('a.png')
objDet.detect(im)
objDet.visual()