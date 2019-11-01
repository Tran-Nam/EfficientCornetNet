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
    smooth_l1_sign = tf.cast(tf.less(diff, sigma), tf.float32) # 0, 1
    smooth_l1_case1 = tf.multiply(diff, diff) / 2.0
    smooth_l1_case2 = tf.subtract(diff, 0.5*sigma) * sigma
    return tf.add(tf.multiply(smooth_l1_sign, smooth_l1_case1), \
        tf.multiply(tf.abs(tf.subtract(smooth_l1_sign, 1.0)), smooth_l1_case2))

def offset_loss(pred, gt, mask):
    n_corner = tf.reduce_sum(mask)  
    # print(n_corner)
    mask = tf.stack((mask, mask), axis=-1) # mask for 2 axis in offset map
    l1_loss = smooth_l1_loss(pred, gt)
    l1_loss /= (n_corner + tf.convert_to_tensor(1e-6, dtype=tf.float32))
    return tf.reduce_sum(tf.multiply(l1_loss, mask))


def focal_loss_2(pred, gt, alpha=2, beta=4):
    zeros = np.zeros_like(pred, dtype=pred.dtype)
    ones = np.ones_like(pred, dtype=pred.dtype)

    # gt<ones <=> negative, so positive coefficent ones-pred
    pos_p_sub = np.where(gt<ones, zeros, ones-pred)

    # gt<ones <=> negative, so negative coefficent is pred
    neg_p_sub = np.where(gt<ones, pred, zeros)

    # neg_p_reduce = array_ops.where(gt==ones, zeros, 1-gt)
    reduce_penalty = ones-gt

    per_entry_loss = -reduce_penalty**beta * (pos_p_sub**alpha * np.log(np.clip(pred, 1e-8, 1.0))\
        + neg_p_sub**alpha * np.log(np.clip(1-pred, 1e-8, 1.0)))

    return np.mean(per_entry_loss)

def focal_loss(pred, gt, alpha=2, beta=4):
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    ones = array_ops.ones_like(pred, dtype=pred.dtype)

    # gt<ones <=> negative, so positive coefficent ones-pred
    pos_p_sub = tf.where(gt<ones, zeros, ones-pred)

    # gt<ones <=> negative, so negative coefficent is pred
    neg_p_sub = tf.where(gt<ones, pred, zeros)

    # neg_p_reduce = array_ops.where(gt==ones, zeros, 1-gt)
    reduce_penalty = ones-gt

    per_entry_loss = -reduce_penalty**beta * (pos_p_sub**alpha * tf.log(tf.clip_by_value(pred, 1e-8, 1.0))\
        + neg_p_sub**alpha * tf.log(tf.clip_by_value(1-pred, 1e-8, 1.0)))

    return tf.reduce_mean(per_entry_loss)

def loss(output, gt):
    heatmap, offset = output 
    heatmap_gt, offset_gt, mask = gt 
    f_loss = focal_loss(heatmap, heatmap_gt)
    o_loss = offset_loss(offset, offset_gt, mask)
    return f_loss + o_loss

# pred = np.random.random((1, 128, 128, 4))
# gt = np.random.random((1, 128, 128, 4))
# pred_off = np.random.random((1, 128, 128, 2))
# gt_off = np.random.random((1, 128, 128, 2))
# mask = np.zeros((1, 128, 128))
# mask[0, 50:60, 60:70] = 1

# a = focal_loss_2(pred, gt)
# b = focal_loss(pred, gt)
# c = smooth_l1_loss(pred, gt)
# d = smooth_l1_loss_2(pred, gt)
# e = offset_loss(pred_off, gt_off, mask)

# print(a)
# with tf.Session() as sess:
#     b, c, d, e = sess.run([b, c, d, e])
#     print(b)
#     print(np.mean(c))
#     print(np.mean(d))
#     print(e)