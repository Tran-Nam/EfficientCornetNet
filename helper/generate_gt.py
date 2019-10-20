import numpy as np 
import cv2 
import utils

def gen_heatmap(im, boxes): # keep aspect ratio
    """
    :param im: origin image
    :param boxes: 4 boxes tl, tr, br, bl: 4*4*2
    """
    # assert boxes.shape[0]==4
    new_im, boxes = utils.resize(im, boxes) 
    side = new_im.shape[0]
    n_boxes = boxes.shape[0]
    heatmap = np.zeros([side, side, n_boxes])
    for i in range(n_boxes):
        center = np.mean(boxes[i, :, :], axis=0).astype('int')
        print(len(center))
        size = utils.getSizePolygon(boxes[i, :, :])
        radius = utils.calcRadius(size)
        utils.draw_gaussian(heatmap[:, :, i], center, radius=radius)
    print(heatmap.shape)
    return new_im, boxes, heatmap

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
    print(boxes.shape)
    print(boxes[0, :, :])

    new_im, boxes, heatmap = gen_heatmap(im, boxes)

    print(boxes.shape)
    for i in range(boxes.shape[1]):
        pt = boxes[0, i, :]
        # print(pt[0, ])
        cv2.circle(new_im, tuple(pt), 2, (0, 0, 255), -1)
    
    heatmap = np.squeeze(heatmap, axis=2)
    max_value = np.max(heatmap)
    heatmap = (heatmap*255/max_value).astype('uint8')
    cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    print(heatmap.shape)
    print(new_im.shape)

    cv2.addWeighted(new_im, 0.1, heatmap, 0.9, 0)
    cv2.imshow('a', new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
