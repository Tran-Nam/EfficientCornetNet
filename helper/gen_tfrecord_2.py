import os
import sys
import pandas as pd
import tensorflow as tf 
import cv2
import numpy as np

from utils import resize

from generate_gt import gen_gt
# from PIL import Image
# from collections import namedtuple, OrderedDict



# from PIL import Image
# from collections import namedtuple, OrderedDict

# flags = tf.app.flags
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('image_dir', '', 'Path to images')
# FLAGS = flags.FLAGS


def _bytes_feature(value):
    """return a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """return a float_list from a float/double"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """returna int64_list from a bool/enum/int/uint"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_label(image_path):
    """
    get label: 4 corners: tl, tr, br, bl
    :param labels: DataFrame label
    :param image_path

    return pts: 4x2 array
    """
    global labels
    pts = np.zeros((4, 2))
    rows = labels[labels['filename']==image_path]

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

    return pts

def create_tfrecord(out, image_dir):
    global labels 
    image_paths = list(set(labels['filename']))
    writer = tf.python_io.TFRecordWriter(out)
    # image_paths = os.listdir(image_dir)
    print('Num image: ', len(image_paths))
    count = 0
    for image_name in image_paths:
        image_path = os.path.join(image_dir, image_name)
        count += 1
        if count%1000==0:
            print('Process: {}/{}'.format(count, len(image_paths)))
            sys.stdout.flush()

        image = load_image(image_path)
        pts = get_label(image_name) ##name

        if image is None:
            continue

        image, pts = resize(image, pts, side=512)

        _, __, heatmap, offset, mask = gen_gt(image, pts)


        # print(count)
        # if count == 100:
        #     cv2.imwrite('a.png', image)
        #     cv2.imwrite('b.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #     print(pts)
        #     break

        feature = {
            'image': _bytes_feature(image.tostring()),
            'heatmap': _bytes_feature(heatmap.tostring()),
            'offset': _bytes_feature(offset.tostring()),
            'mask': _bytes_feature(mask.tostring())

            # 'label': _bytes_feature(pts.tostring())
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

if __name__=='__main__':
    labels_csv = '../../data/labels.csv'
    image_dir = '../../images'
    labels = pd.read_csv(labels_csv)
    out = 'data_2.tfrecords'

    #out = 'data.tfrecords'
    create_tfrecord(out, image_dir)

