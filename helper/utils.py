import numpy as np 
import cv2

def calcLineLength(point1, point2):
    x1, y1 = point1 
    x2, y2 = point2 
    length = ((x1-x2)**2+(y1-y2)**2)**0.5
    return length 

def orderPoints(pts):
    # order: tl, tr, br, bl
    pts = np.array(pts).reshape(4, 2)
    # rect = np.zeros((4, 2), dtype='float32')
    tl_idx = np.argmin(np.sum(pts, axis=1))
    br_idx = np.argmax(np.sum(pts, axis=1))
    tr_idx = np.argmin(np.diff(pts, axis=1))
    bl_idx = np.argmax(np.diff(pts, axis=1))
    rect_idx = [tl_idx, tr_idx, br_idx, bl_idx]
    return pts[rect_idx]
    
def calcRadius(size, iou_thresh=0.7): # implement base on origin paper
    h, w = size
    l = (h**2+w**2)**0.5

    # det inside
    a1 = 4
    b1 = -4*l
    c1 = -(1-iou_thresh)*l**2
    delta1 = b1**2-4*a1*c1
    r1 = (-b1 + delta1**0.5) / (2*a1)

    # det outside
    a2 = 4
    b2 = 4*l
    c2 = (1-1/iou_thresh)*l**2
    delta2 = b2**2 - 4*a2*c2
    r2 = (-b2 + delta2**0.5) / (2*a2)

    # det cross
    r3 = (((1-iou_thresh)*l**2)/4)**0.5

    return int(min(r1, r2, r3))

def gaussian2D(size, sigma=1):
    h, w = size
    gaussian = np.zeros((h, w))
    center = [h//2, w//2]
    for i in range(h):
        for j in range(w):
            gaussian[i, j] = 1 / (2*np.pi*sigma**2) * np.exp(-((i-center[0])**2+(j-center[1])**2)/(2*sigma**2))
    # print(gaussian.shape)
    gaussian = gaussian / np.max(gaussian)
    return gaussian

def draw_gaussian(heatmap, center, radius):
    diameter = 2*radius + 1
    sigma = diameter / 6 # in paper
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    center_x, center_y = center
    h, w = heatmap.shape[0:2]
    
    left = min(center_x, radius)
    right = min(w-center_x, radius+1)
    top = min(center_y, radius)
    bottom = min(h-center_y, radius+1)

    # print(heatmap[:5, :5])
    mask_gaussian = gaussian[radius-top: radius+bottom, radius-left: radius+right]
    mask_heatmap = heatmap[center_y-top: center_y+bottom, center_x-left: center_x+right]
    np.maximum(mask_heatmap, mask_gaussian, out=mask_heatmap)
    # print(heatmap.shape)
    # print(mask_heatmap)
    # print(heatmap[:5, :5])

def gaussian2D_other(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # print(h.shape)
    return h

def resize(im, boxes, side=128): # resize image keep aspect ratio, padding
    """
    :param im: origin image
    :param boxes: boxes correspond tl, tr, br, bl: n_boxes*4*2
    :param side: new image size
    return im and boxes after resize
    """
    im_h, im_w = im.shape[:2]
    ratio = side / max(im_h, im_w)
    new_size = (int(im_h*ratio), int(im_w*ratio))
    new_im = cv2.resize(im, None, fx=ratio, fy=ratio)
    boxes = boxes.astype('float32')
    boxes[:, :, 0] *= ratio
    boxes[:, :, 1] *= ratio

    big_image = np.ones([side, side, 3], dtype='float32')*np.mean(new_im)
    padding_y = (side - new_im.shape[0])//2
    padding_x = (side - new_im.shape[1])//2
    big_image[padding_y: padding_y+new_im.shape[0],
        padding_x: padding_x+new_im.shape[1]] = new_im
    big_image = big_image.astype('uint8')

    boxes[:, :, 0] += padding_x
    boxes[:, :, 1] += padding_y
    boxes = boxes.astype('int')

    return big_image, boxes
    
def getSizePolygon(pts):
    pts = orderPoints(pts)
    tl, tr, br, bl = pts 
    side_left = calcLineLength(tl, bl)
    side_top = calcLineLength(tl, tr)
    side_right = calcLineLength(tr, br)
    side_bottom = calcLineLength(br, bl)
    size = (max(side_left, side_right), max(side_top, side_bottom))
    return size

if __name__ == '__main__':
    import cv2
    im = np.zeros((100, 100, 3))
    pts = [[20, 20],
        [60, 60], 
        [80, 20],
        [30, 80]]
    for pt in pts:
        cv2.circle(im, tuple(pt), 2, (0, 0, 255), -1)

    sigma = 2
    size = (50, 50)
    heatmap = gaussian2D(size, sigma=sigma)
    h = gaussian2D_other(size, sigma=sigma)

    diff = heatmap - h
    diff[diff<1] = 0
    # print(np.where(diff>0))

    heatmap = np.zeros((50, 50))
    draw_gaussian(heatmap, (1, 1), 2)
    
    # cv2.imshow('a', heatmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
