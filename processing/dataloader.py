import tensorflow as tf 
import pandas as pd 
import numpy as np 
import os
import pathlib 
from PIL import Image
import cv2
# import matplotlib.pyplot as plt 
tf.enable_eager_execution()

import sys
sys.path.append('..')

# from .augment import random_crop, rotate, add_noise, color_variance
from helper.generate_gt import gen_gt
from helper.utils import resize

class DataLoader(object):
    def __init__(self, image_dir, labels_path='../../data/labels.csv'):
        # tf.enable_eager_execution()
        self.image_dir = pathlib.Path(image_dir)
        self.labels = pd.read_csv(labels_path)
        # self.ext = ['jpg', 'png']
        self.ds = tf.data.Dataset.list_files(os.path.join(image_dir, '*.jpg'))
        n_img = len(list(set(self.labels['filename'])))
        print('No. images: {}'.format(n_img))

        self.BATCH_SIZE = 4
        self.IMG_HEIGHT = 512
        self.IMG_WIDTH = 512
        self.STEP_PER_EPOCH = np.ceil(n_img/self.BATCH_SIZE)

    def get_corners(self, file_path):
        """
        get corners by file path
        """
        # print(file_path)
        # print(type(file_path))
        file_path = file_path.decode('utf-8')
        # print(file_path.decode('utf-8'))
        # filename = tf.string_split([file_path], '/').values[-1]
        # filename = str(filename)
        filename = str(file_path).split('/')[-1]
        # print(filename)
        pts = np.zeros((4, 2))
        rows = self.labels[self.labels['filename']==filename]
        # print(rows)
        for idx, row in rows.iterrows():
            row_c = row['class']
            if row_c=='topleft':
                pts_loc = 0
            elif row_c=='topright':
                pts_loc = 1
            elif row_c=='bottomright':
                pts_loc = 2
            elif row_c=='bottomleft':
                pts_loc = 3
            pts[pts_loc] = np.array([row['x'], row['y']])
        # pts = np.array(pts).reshape(4, 2)
        return pts

    def get_labels(self, file_path):
        """
        generate heatmap and offset
        """
        pts = self.get_corners(file_path)
        pass

    @staticmethod
    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        # print(img.get_shape().as_list())
        # print('*'*30)
        return img

    def process_path(self, file_path):
        # heatmap, offset = self.get_labels(file_path)
        # img = tf.io.read_file(file_path)
        # img = self.decode_img(img)
        # file_path = tf.strings.as_string(file_path)
        # print(file_path)
        # print('-'*20)
        img = Image.open(file_path)
        img = np.array(img)
        # print(img.shape)
        # input()
        # print(img.get_shape().as_list())

        #################################
        pts = self.get_corners(file_path)
        # print(pts)
        # resize to 512x512
        img, pts = resize(img, pts, side=512)
        _, __, heatmap, offset = gen_gt(img, pts)
        # print('Gen heatmap: Done!')
        # print(pts)
        # print(np.where(heatmap>0)[0])
        # print('*'*30)
        return img, heatmap, offset

    def create_pairs(self):
        self.labeled_ds = self.ds.map(lambda x: tf.py_func(self.process_path, [x], [tf.uint8, tf.float32, tf.float32]), num_parallel_calls=4) #num cores
        self.train_ds = self.prepare_for_training(self.labeled_ds)
        return self

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=32):       
        # self.create_pairs()
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        
        ds = ds.shuffle(buffer_size=shuffle_buffer_size) # num batch per epoch

        # repeat forever
        ds = ds.repeat()
        ds = ds.batch(self.BATCH_SIZE)

        # let dataset fetch batch in background when model is training
        ds = ds.prefetch(buffer_size=32)

        return ds

    def next_batch(self):
        image_batch, heatmap_batch, offset_batch = next(iter(self.train_ds))
        return image_batch, heatmap_batch, offset_batch

if __name__=='__main__':
    dataload = DataLoader('../../images')
    dataload.create_pairs()
    # for i in range(10):
    #     image_batch, heatmap_batch, offset_batch = dataload.next_batch()
    #     print(image_batch.shape, heatmap_batch.shape, offset_batch.shape)

    import time
    default_timeit_steps = 1000
    def timeit(ds, steps=default_timeit_steps):
        start = time.time()
        it = iter(ds)
        for i in range(steps):
            batch = next(it)
            if i%10==0:
                print('.', end='')
        print()
        end = time.time()
        duration = end - start
        print('{} batches: {} s'.format(steps, duration))
        print('{:0.5f} Images/s'.format(4*steps/duration))

    timeit(dataload.ds)
    # for i in range(image_batch.shape[0]):   
    #     dir_name = str(i)
    #     if not os.path.isdir(dir_name):
    #         os.mkdir(dir_name)
    #     im = image_batch[i, :, :, :].numpy()   
    #     heatmap = heatmap_batch[i, :, :, :].numpy()
    #     offset = offset_batch[i, :, :, :].numpy()

    #     cv2.imwrite(os.path.join(dir_name, 'im.png'), im.astype('uint8'))

    #     max_value = np.max(heatmap)
    #     heatmap = (heatmap*255/max_value).astype('uint8')
    #     for j in range(heatmap.shape[2]):
    #         im_ = cv2.applyColorMap(heatmap[:, :, j], cv2.COLORMAP_JET)
    #         cv2.imwrite(os.path.join(dir_name, '{}.png'.format(j)), im_)

    #     max_value = np.max(offset)
    #     offset = (offset*255/(max_value+1e-8)).astype('uint8')
    #     for j in range(offset.shape[2]):
    #         im_ = cv2.applyColorMap(offset[:, :, j], cv2.COLORMAP_JET)
    #         cv2.imwrite(os.path.join(dir_name, '{}_.png'.format(j)), im_)