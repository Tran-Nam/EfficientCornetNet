import numpy as np 
import cv2 
import utils

def order_point(polygon):
    polygon = np.array(polygon).reshape(4, 2)
    tl_idx = np.argmin(np.sum(polygon, axis=1))
    br_idx = np.argmax(np.sum(polygon, axis=1))
    tl = polygon[tl_idx]
    br = polygon[br_idx]
    
    tmp = np.delete(polygon, [tl_idx, br_idx], axis=0)
    res0 = tmp[0][0] - tmp[0][1]
    res1 = tmp[1][0] - tmp[1][1]
    if res0 > res1:
        tr = tmp[0]
        bl = tmp[1]
    else:
        tr = tmp[1]
        bl = tmp[0]
    
    ordered_polygon = [tl, tr, br, bl]
    return ordered_polygon


def get_size_polygon(polygon):
    tl, tr, br, bl = polygon 
    top_w = utils.calc_line_length(tr, tl)
    bottom_w = utils.calc_line_length(br, bl)
    left_h = utils.calc_line_length(bl, tl)
    right_h = utils.calc_line_length(br, tr)
    w = int(max(top_w, bottom_w))
    h = int(max(left_h, right_h))
    return (w, h)

def warp_rect(polygon):
    """
    warp polygon to rectangle
    width, height of rect is greater value in polygon

    return
    w, h, rect
    """
    w, h = get_size_polygon(polygon)

    rect_pattern = [[0, 0],
                [w, 0],
                [w, h],
                [0, h]]
    
    polygon = polygon.astype('float32')
    rect_pattern = np.array(rect_pattern).astype('float32')
    # print(polygon)
    # print(rect_pattern)
    M = cv2.getPerspectiveTransform(polygon, rect_pattern, cv2.CV_16S)

    rect = cv2.warpPerspective(polygon, M, (w, h))

    return w, h, rect

def getRadius(size, iou_thresh=0.7):
    w, h = size
    l = (w**2+h**2)**0.5 #diagnoal line
    # 3 case
    # detection inside
    a1 = 4*w*h/(l**2)
    b1 = -4*w*h/l 
    c1 = -(1-iou_thresh)*w*h
    delta1 = b1**2-4*a*c
    r1 = (-b1+delta1**0.5)/(2*a1)

    # detection outside
    

    # detection cross


    pass

if __name__=='__main__':
    polygon = [[1, 1],
            [3, 2],
            [4, 5],
            [0, 2]]
    polygon = np.array(polygon)
    w, h, rect = warp_rect(polygon)
    print(polygon)
    print(rect)