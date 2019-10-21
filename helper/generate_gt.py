import numpy as np 
import cv2 
import utils

def gen_heatmap(im, boxes): # keep aspect ratio
    
    """
    :param im: origin image
    :param boxes: 4 boxes tl, tr, br, bl: 4*4*2
    """
    # assert boxes.shape[0]==4
    im, boxes = utils.resize(im, boxes) # resize keep aspect ratio
    # pts = np.mean(boxes, axis=1) # pts: 4*2
    
    pts =  np.squeeze(boxes, axis=0) # DEBUG
    # print(pts.shape)
    # pts = np.squeeze(pts, 1)
    side = im.shape[0]
    n_pt = boxes.shape[1]
    heatmap = np.zeros([side, side, n_pt])
    size = utils.getSizePolygon(pts)
    radius = utils.calcRadius(size)
    print(radius, size)
    for i in range(n_pt):
        center = pts[i]
        # print(len(center))
        
        utils.draw_gaussian(heatmap[:, :, i], center, radius=radius)
    # print(heatmap.shape)
    return im, boxes, heatmap

if __name__=='__main__':
    # im = np.random.random((100, 100, 3))
    im = np.zeros([100, 100, 3])
    # im = (im*255).astype('uint8')

    tl = [10, 10]
    tr = [70, 25]
    br = [80, 80]
    bl = [20, 70]

    _boxes = [tl, tr, br, bl]
    boxes = [tl, tr, br, bl]
    boxes = np.array(boxes).reshape(-1, 4, 2)
    # print(boxes.shape)
    # print(boxes[0, :, :])

    new_im, boxes, heatmap = gen_heatmap(im, boxes)
    print('-'*20)
    print(heatmap.shape)

    # print(boxes.shape)
    for i in range(boxes.shape[1]):
        pt = boxes[0, i, :]
        # print(pt[0, ])
        cv2.circle(new_im, tuple(pt), 1, (0, 0, 255), -1)
    
    # heatmap = np.squeeze(heatmap, axis=2)
    max_value = np.max(heatmap)
    heatmap = (heatmap*255/max_value).astype('uint8')
    # cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # print(heatmap.shape)
    # print(new_im.shape)

    # cv2.imwrite('a.png', new_im)
    # for i in range(heatmap.shape[2]):
    #     im_ = cv2.applyColorMap(heatmap[:, :, i], cv2.COLORMAP_JET)
    #     cv2.imwrite('{}.png'.format(i), im_)
    # cv2.addWeighted(new_im, 0.1, heatmap, 0.9, 0)
    # cv2.imshow('a', new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
