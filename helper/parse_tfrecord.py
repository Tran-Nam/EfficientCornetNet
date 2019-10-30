import tensorflow as tf
import numpy as np 
import cv2 
import sys 
import os 

# tf.enable_eager_execution()

def parser(record):
    keys_to_features = {
        'image': tf.FixedLenFeature([], tf.string),
        'heatmap': tf.FixedLenFeature([], tf.string),
        'offset': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string)
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image, tf.float32) *1./255
    image = tf.reshape(image, [512, 512, 3])
    heatmap = tf.decode_raw(parsed['heatmap'], tf.float32)
    heatmap = tf.reshape(heatmap, [128, 128, 4])
    offset = tf.decode_raw(parsed['offset'], tf.float32)
    offset = tf.reshape(offset, [128, 128, 2])
    mask = tf.decode_raw(parsed['mask'], tf.float32)
    mask = tf.reshape(mask, [128, 128])
    # pts = tf.cast(pts, tf.float32)

    return {'image': image,
         'heatmap': heatmap, 
         'offset': offset,
         'mask': mask}

def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=3)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(parser, 32)
    )
    dataset = dataset.prefetch(buffer_size=2)
    print(dataset)
    return dataset

def train_input_fn():
    return input_fn(filenames=['data_2.tfrecords'])

dataset = train_input_fn()
# i = 0
# for data in dataset:
#     i += 1
#     image = data['image'].numpy()
#     print(image.shape)
# print(i*32)

