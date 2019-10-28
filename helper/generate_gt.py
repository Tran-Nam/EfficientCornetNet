import numpy as np 
import cv2 
from . import utils
# import utils

def gen_gt(im, pts, side=128): # keep aspect ratio   
    """
    :param im: origin image
    :param boxes: 4 pt tl, tr, br, bl: 4*2
    """
    # assert pts.shape[0]==4

    ratio = max(im.shape[0], im.shape[1]) / side 
    # ratio = 4

    # gen offset map
    offset = np.zeros([side, side, 2])  
    for i in range(pts.shape[0]):
        corner_loc = pts[i]
        corner_loc_map_raw = corner_loc / ratio
        corner_loc_map = np.floor(corner_loc_map_raw)
        offset_value = corner_loc_map_raw - corner_loc_map
        offset[int(corner_loc_map[1]), int(corner_loc_map[0]), :] = offset_value
    # print(pts)

    im, pts = utils.resize(im, pts, side=side) # resize keep aspect ratio

    # side = im.shape[0]
    n_pt = pts.shape[0]

    # gen heatmap
    heatmap = np.zeros([side, side, n_pt])
    mask = np.zeros([side, side])
    size = utils.getSizePolygon(pts)
    radius = utils.calcRadius(size)
    # print(radius, size)
    for i in range(n_pt):
        center = pts[i]     
        mask[center[0], center[1]] = 1
        utils.draw_gaussian(heatmap[:, :, i], center, radius=radius)
    
    heatmap = heatmap.astype('float32')
    offset = offset.astype('float32')
    mask = mask.astype('float32')

    return im, pts, heatmap, offset, mask
 

if __name__=='__main__':
    # im = np.random.random((100, 100, 3))
    im = np.zeros([512, 512, 3])
    # im = (im*255).astype('uint8')

    tl = [100, 101]
    tr = [500, 150]
    br = [400, 410]
    bl = [60, 450]

    pts = [tl, tr, br, bl]
    pts = np.array(pts).reshape(4, 2)
    # print(boxes.shape)
    # print(boxes[0, :, :])

    new_im, new_pts, heatmap, offset, mask = gen_gt(im, pts)
    
    print('-'*20)
    print(heatmap.shape)
    print(offset.shape)
    print(mask.shape)
    print(np.where(mask==1))
    

    # print(boxes.shape)
    for i in range(new_pts.shape[0]):
        pt = new_pts[i]
        # print(pt[0, ])
        cv2.circle(new_im, tuple(pt), 1, (0, 0, 255), -1)

    for i in range(pts.shape[0]):
        pt = pts[i]
        # print(pt[0, ])
        cv2.circle(im, tuple(pt), 1, (0, 0, 255), -1)
    
    # heatmap = np.squeeze(heatmap, axis=2)
    max_value = np.max(heatmap)
    heatmap = (heatmap*255/max_value).astype('uint8')

    max_value_ = np.max(offset)
    offset = (offset*255/max_value_).astype('uint8')
    offset[offset>0]=100
    # cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # print(heatmap.shape)
    # print(new_im.shape)
    cv2.imwrite('b.png', im)
    cv2.imwrite('a.png', new_im)
    for i in range(heatmap.shape[2]):
        im_ = cv2.applyColorMap(heatmap[:, :, i], cv2.COLORMAP_JET)
        cv2.imwrite('{}.png'.format(i), im_)

    for i in range(offset.shape[2]):
        # ret, map_ = cv2.THRESH_BINARY(offset[:, :, i], 1, 255, cv2.THRESH_BINARY)
        im_ = cv2.applyColorMap(offset[:, :, i], cv2.COLORMAP_JET)
        # ret, im_ = cv2.THRESH_BINARY(im_, 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite('{}_.png'.format(i), im_)
    # cv2.addWeighted(new_im, 0.1, heatmap, 0.9, 0)
    # cv2.imshow('a', new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
