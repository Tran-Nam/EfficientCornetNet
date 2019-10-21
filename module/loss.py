import numpy as np 
import tensorflow as tf
from tensorflow.python.ops import array_ops

def smooth_l1_loss(pred, gt, sigma=1):
    pass

def offset_loss(offset_pred, offset_gt, mask):
    pass

def focal_loss(pred, gt, alpha=2, beta=4):
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    ones = array_ops.ones_like(pred, dtype=pred.dtype)

    # gt<ones <=> negative, so positive coefficent ones-pred
    pos_p_sub = array_ops.where(gt<ones, zeros, ones-pred)

    # gt<ones <=> negative, so negative coefficent is pred
    neg_p_sub = array_ops.where(gt<ones, pred, zeros)

    # neg_p_reduce = array_ops.where(gt==ones, zeros, 1-gt)
    reduce_penalty = ones-gt

    per_entry_loss = -reduce_penalty**beta * (pos_p_sub**alpha * tf.log(tf.clip_by_value(pred, 1e-8, 1.0))\
        + neg_p_sub**alpha * tf.log(tf.clip_by_value(1-pred, 1e-8, 1.0)))

    return tf.reduce_mean(per_entry_loss)