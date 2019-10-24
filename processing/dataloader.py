import tensorflow as tf 
import pandas as pd 
import numpy as np 
import os

import sys
sys.path.append('..')

from .augment import random_crop, rotate, add_noise, color_variance
from helper.generate_gt import gen_gt
from helper.utils import resize

class DataLoader(object):
    def __init__(self, image_dir, labels_path='../../data/labels.csv'):
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_path)
        # self.ext = ['jpg', 'png']
        self.ds = tf.data.Dataset.list_files(os.path.join(image_dir, '*'))
        n_img = len(list(set(self.ds['filename'])))

        self.BATCH_SIZE = 4
        self.IMG_HEIGHT = 512
        self.IMG_WIDTH = 512
        self.STEP_PER_EPOCH = np.ceil(n_img/self.BATCH_SIZE)

    def get_corners(self, file_path):
        """
        get corners by file path
        """
        filename = file_path.split('/')[-1]
        pts = np.empty((4, 2))
        rows = self.labels[self.labels['filename']==filename]
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
            pts[pts_loc] = np.array([row['x'], row['y']]).reshape[1, 2]
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
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def process_path(self, file_path):
        # heatmap, offset = self.get_labels(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        #################################
        pts = self.get_corners(file_path)
        # resize to 512x512
        img, pts = resize(img, pts, side=512)
        _, __, heatmap, offset = gen_gt(img, pts)


        return img, heatmap, offset

    def create_pairs(self):
        self.labeled_ds = self.ds.map(process_path, num_parallel_calls=AUTOTUNE)
        self.train_ds = self.prepare_for_training(self.labeled_ds)
        return self

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):       
        # self.create_pairs()
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # repeat forever
        ds = ds.repeat()
        ds = ds.batch(self.BATCH_SIZE)

        # let dataset fetch batch in background when model is training
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def next_batch(self):
        image_batch, label_batch = next(iter(self.train_ds))
        return image_batch, label_batch

if __name__=='__main__':
    dataload = DataLoader('../../data/images')
    dataload.create_pairs()
    image_batch, label_batch = dataload.next_batch()
    print(image_batch.shape)