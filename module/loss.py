import numpy as np 
import tensorflow as tf
from tensorflow.python.ops import array_ops

def smooth_l1_loss_2(pred,targets,sigma=1):
    # diff = pred -targets
    # abs_diff = tf.abs(diff)
    # smoothL1_sign =tf.to_float(tf.less(abs_diff, 1))
    # loss = tf.pow(diff, 2) * 0.5 * smoothL1_sign + (abs_diff - 0.5) * (1. - smoothL1_sign)
    # return loss
    sigma2 = sigma * sigma

    diff = tf.subtract(pred, targets)

    smooth_l1_sign = tf.cast(tf.less(tf.abs(diff), 1.0 / sigma2), tf.float64)
    smooth_l1_option1 = tf.multiply(tf.multiply(diff, diff), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(diff), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
    return smooth_l1_result

def smooth_l1_loss(pred, gt, sigma=1):
    diff = tf.abs(tf.subtract(pred, gt))
    # zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    # ones = array_ops.ones_like(pred, dtype=pred.dtype)
    # mask = tf.where()
    smooth_l1_sign = tf.cast(tf.less(diff, sigma), tf.float64) # 0, 1
    smooth_l1_case1 = tf.multiply(diff, diff) / 2.0
    smooth_l1_case2 = tf.subtract(diff, 0.5*sigma) * sigma
    return tf.add(tf.multiply(smooth_l1_sign, smooth_l1_case1), \
        tf.multiply(tf.abs(tf.subtract(smooth_l1_sign, 1.0)), smooth_l1_case2))

def offset_loss(pred, gt, mask):
    n_corner = tf.reduce_sum(mask)  
    # print(n_corner)
    mask = tf.stack((mask, mask), axis=-1) # mask for 2 axis in offset map
    l1_loss = smooth_l1_loss(pred, gt)
    l1_loss /= (n_corner + tf.convert_to_tensor(1e-6, dtype=tf.float64))
    return tf.reduce_sum(tf.multiply(l1_loss, mask))


def focal_loss_2(pred, gt, alpha=2, beta=4):
    zeros = np.zeros_like(pred, dtype=pred.dtype)
    ones = np.ones_like(pred, dtype=pred.dtype)

    # gt<ones <=> negative, so positive coefficent ones-pred
    # pos_p_sub = np.where(gt<ones, zeros, ones-pred)
    pos_p_sub = np.where(gt==ones, ones-pred, zeros)

    # gt<ones <=> negative, so negative coefficent is pred
    # neg_p_sub = np.where(gt<ones, pred, zeros)
    neg_p_sub = np.where(gt<ones, pred, zeros)

    # neg_p_reduce = array_ops.where(gt==ones, zeros, 1-gt)
    reduce_penalty = ones-gt

    per_entry_loss = -reduce_penalty**beta * (pos_p_sub**alpha * np.log(np.clip(pred, 1e-8, 1.0))\
        + neg_p_sub**alpha * np.log(np.clip(1-pred, 1e-8, 1.0)))

    return np.mean(per_entry_loss)

def focal_loss(pred, gt, alpha=2, beta=4):
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    ones = array_ops.ones_like(pred, dtype=pred.dtype)
    gt = tf.cast(gt, dtype=pred.dtype)

    # gt<ones <=> negative, so positive coefficent ones-pred
    # pos_p_sub = tf.where(gt<ones, zeros, ones-pred)
    pos_p_sub = tf.where(tf.equal(gt, 1), ones-pred, zeros)

    # gt<ones <=> negative, so negative coefficent is pred
    # neg_p_sub = tf.where(gt<ones, pred, zeros)
    neg_p_sub = tf.where(tf.less(gt, 1), pred, zeros)

    # neg_p_reduce = array_ops.where(gt==ones, zeros, 1-gt)
    reduce_penalty = ones-gt

    # per_entry_loss = -reduce_penalty**beta * (pos_p_sub**alpha * tf.log(tf.clip_by_value(pred, 1e-8, 1.0))\
    #     + neg_p_sub**alpha * tf.log(tf.clip_by_value(1-pred, 1e-8, 1.0))) #############
    per_entry_loss = -(pos_p_sub**alpha * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
        + reduce_penalty**beta * neg_p_sub**alpha * tf.log(tf.clip_by_value(1-pred, 1e-8, 1.0)))

    # max_pred, min_pred = tf.reduce_max(pred), tf.reduce_min(pred)
    # max_gt, min_gt = tf.reduce_max(gt), tf.reduce_min(gt)
    

    return tf.reduce_sum(per_entry_loss)#, (pos_p_sub, neg_p_sub, min_pred, max_pred, min_gt, max_gt)

def focal_loss_3(preds,gt):
    # print(gt.get_shape().as_list())
    zeros=tf.zeros_like(gt)
    ones=tf.ones_like(gt)
    num_pos=tf.reduce_sum(tf.where(tf.equal(gt,1),ones,zeros))
    print(num_pos)
    # num_pos = tf.cast(num_pos, dtype=tf.float32)
    loss=0
    # loss = tf.cast(loss, tf.float32)
    #loss=tf.reduce_mean(tf.log(preds))
    for pre in preds:
        pre = tf.expand_dims(pre, axis=0)
        pos_weight=tf.where(tf.equal(gt,1),ones-pre,zeros)
        neg_weight=tf.where(tf.less(gt,1),pre,zeros)
        pos_loss=tf.reduce_sum(tf.log(tf.clip_by_value(pre, 1e-8, 1.0)) * tf.pow(pos_weight,2))
        neg_loss=tf.reduce_sum(tf.pow((1-gt),4)*tf.pow(neg_weight,2)*tf.log(tf.clip_by_value(1-pre, 1e-8, 1.0)))
        # print(pos_loss, neg_loss)
        print(num_pos.dtype, pos_loss.dtype, neg_loss.dtype)
        loss=loss-(pos_loss+neg_loss)#/(num_pos+tf.convert_to_tensor(1e-8, dtype=tf.float64))
    return loss, (pos_loss, neg_loss)

def focal_loss_4(pred, gt):
    # num_pos=tf.reduce_sum(tf.where(tf.equal(gt,1),ones,zeros))
    zeros = tf.zeros_like(pred, dtype=pred.dtype)
    ones = tf.ones_like(pred, dtype=pred.dtype)
    num_pos=tf.reduce_sum(tf.where(tf.equal(gt,1),ones,zeros))
    # print(num_pos)
    pos_weight = tf.where(tf.equal(gt, 1), ones-pred, zeros)
    neg_weight = tf.where(tf.less(gt, 1), pred, zeros)
    pos_loss = tf.reduce_sum(tf.log(tf.clip_by_value(pred, 1e-8, 1.0))*tf.pow(pos_weight, 2))
    neg_loss = tf.reduce_sum(tf.pow((1-gt), 4)*tf.pow(neg_weight, 2)*tf.log(tf.clip_by_value(1-pred, 1e-8, 1.0)))
    loss = -(pos_loss+neg_loss)/num_pos
    return loss, num_pos

def loss(output, gt):
    heatmap, offset = output 
    heatmap_gt, offset_gt, mask = gt 
    f_loss = focal_loss(heatmap, heatmap_gt)
    o_loss = offset_loss(offset, offset_gt, mask)
    return f_loss + o_loss


if __name__=='__main__':
    import cv2
    import sys
    sys.path.append('..')
    from helper.generate_gt import gen_gt

    im = np.zeros((512, 512, 3))
    pts = np.array(
        [[50, 20],
        [300, 50], 
        [270, 360],
        [40, 300]]
    )
    for i in range(pts.shape[0]):
        pt = pts[i]
        cv2.circle(im, tuple(pt), 5, (0, 0, 255), -1)
    
    new_im, new_pts, heatmap, offset, mask = gen_gt(im, pts)
    heatmpa = heatmap/np.max(heatmap) # normalize 0-1

    heatmap_gt = np.expand_dims(heatmap, axis=0)
    pred_1 = heatmap_gt
    pred_2 = np.zeros(heatmap_gt.shape) 
    pred_3 = np.ones(heatmap_gt.shape)
    pred_4 = np.random.random(heatmap_gt.shape)
    loss_1 = focal_loss(pred_1, heatmap_gt)
    loss_2 = focal_loss(pred_2, heatmap_gt)
    loss_3 = focal_loss(pred_3, heatmap_gt)
    loss_4 = focal_loss(pred_4, heatmap_gt)
    print('-'*50)
    # print(np.where(heatmap_gt==1))
    with tf.Session() as sess:
        a, b, c, d = sess.run([loss_1, loss_2, loss_3, loss_4])
        print('Loss of 2 ground truth (expect approx 0): ', a)
        print('Loss of zeros with ground truth (expect >> 0): ', b)
        print('Loss of ones with ground truth (expect >> 0): ', c)
        print('Loss of random with ground truth (expect >> 0): ', d)
        # pos, neg, min_pred, max_pred, min_gt, max_gt = sub
        # # print(pred.shape)
        # pos = np.squeeze(pos, axis=0)
        # neg = np.squeeze(neg, axis=0)

        # print(min_pred, max_pred, min_gt, max_gt)
        # print(np.max(pos[:, :, 0]), np.min(pos[:, :, 0]))
        # # print(np.max(pred[:, :, 0]), np.min(pred[:, :, 0]))
        # # print(np.max(gt[:, :, 0]), np.min(gt[:, :, 0]))
        # print(np.max(neg[:, :, 0]), np.min(neg[:, :, 0]))

        # cv2.imwrite('a.png', pos[:, :, 0])
        # cv2.imwrite('b.png', neg[:, :, 0])
        # pos_coef = cv2.applyColorMap((pos[:, :, 0]/np.max(pos[:, :, 0])*255).astype('uint8'), cv2.COLORMAP_JET)
        # neg_coef = cv2.applyColorMap((neg[:, :, 0]/np.max(neg[:, :, 0])*255).astype('uint8'), cv2.COLORMAP_JET)
        # cv2.imshow('b', pos_coef)
        # # cv2.imshow('c', pos[:, :, 1])
        # # cv2.imshow('d', pos[:, :, 3])
        # cv2.imshow('a', neg_coef)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
       

        # cv2.imshow('a', pos)



    # cv2.imshow('b', heatmap[:, :, 0])
    # cv2.imshow('c', heatmap[:, :, 1])
    # cv2.imshow('d', heatmap[:, :, 3])
    # cv2.imshow('a', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    """
    pred = np.random.random((1, 128, 128, 4))
    pred = pred / np.max(pred)
    # gt = np.random.random((1, 128, 128, 4))
    gt = pred
    pred_off = np.random.random((1, 128, 128, 2))
    # gt_off = np.random.random((1, 128, 128, 2))
    gt_off = pred_off
    mask = np.zeros((1, 128, 128))
    mask[:, 50:60, 60:70] = 1

    # a = focal_loss_2(pred, gt)
    b = focal_loss(pred, gt)
    # f, t = focal_loss_3(pred, gt)
    i, j = focal_loss_4(pred, gt)
    c = smooth_l1_loss(pred_off, gt_off)
    d = smooth_l1_loss_2(pred_off, gt_off)
    e = offset_loss(pred_off, gt_off, mask)

    # print(a)
    with tf.Session() as sess:
        b, i, j, c, d, e = sess.run([b, i, j, c, d, e])
        print(b)
        # print(f, t)
        # print(i, j)
        # print(np.mean(c))
        # print(np.mean(d))
        # print(e)
    """