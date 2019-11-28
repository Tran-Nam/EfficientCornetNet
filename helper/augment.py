import numpy as np 
import imutils 
from utils import findMax2d

def randomCrop(image, pts):
    """
    random crop image, remain area contain object
    :param image: origin image
    :param pts: coordinate 4 corner, shape 4x2
    """
    h, w = image.shape[:2]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    x_begin = np.random.randint(0, x_min)
    y_begin = np.random.randint(0, y_min)
    x_end = np.random.randnit(x_max, w)
    y_end = np.random.randint(y_max, h)
    new_im = image[y_begin: y_end, x_begin: x_end, :]
    new_pts = pts - np.array([x_begin, y_begin])
    return new_im, new_pts 

def rotate(image, pts, angle):
    """
    rotate image, refine coordinate of pts
    """
    padding_value = np.mean(image)
    new_im = imutils.rotate_bound(image, angle)
    new_im[new_im==0] = padding_value
    new_pts = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        x, y = pts[i]
        im_tmp = np.zeros(image.shape[:2])
        im_tmp[y, x] = 255
        im_tmp_rotate = imutils.rotate_bound(im_tmp, angle)
        m, n = findMax2d(im_tmp_rotate)
        new_pt = np.array([n, m])
        new_pts[i] = new_pt
    return new_im, new_pts

if __name__=='__main__':
    import cv2
    im_path = '../image_data/b.jpg'
    im = cv2.imread(im_path)
    pts = np.array([
        [100, 100],
        [500, 500],
        [900, 900],
        [600, 600]
    ])
    new_im, new_pts = rotate(im, pts, 30)
    for i in range(pts.shape[0]):
        cv2.circle(im, tuple(pts[i]), 50, (0, 0, 255), -1)

    
    for i in range(new_pts.shape[0]):
        cv2.circle(new_im, tuple(new_pts[i]), 50, (0, 0, 255), -1)

    cv2.imwrite('c.png', im)
    cv2.imwrite('d.png', new_im)

    # a = (np.ones((50, 60))*50).astype('uint8')
    # cv2.circle(a, (4, 4), 2, (0, 0, 255), -1)
    # a[4, 4] = 255
    # b = imutils.rotate_bound(a, 30)
    # i, j = findMax2d(b)
    # cv2.circle(b, (j, i), 2, (0, 0, 255), -1)
    # print(j, i)

    # cv2.imwrite('a.png', a)
    # cv2.imwrite('b.png', b)
    
