import os
import io
import pandas as pd
import tensorflow as tf 

from PIL import Image
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

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


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) \
        for filename, x in zip(gb.groups.keys(), gb.groups)]

def class_text_to_int(row_label):
    if row_label == 'topleft':
        return 1
    if row_label == 'topright':
        return 2
    if row_label == 'bottomright':
        return 3
    if row_label == 'bottomleft':
        return 4
    else:
        None

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    center_x = []
    center_y = []
    # xmins = []
    # xmaxs = []
    # ymins = []
    # ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        # xmins.append(row['xmin'] / width)
        # xmaxs.append(row['xmax'] / width)
        # ymins.append(row['ymin'] / height)
        # ymaxs.append(row['ymax'] / height)
        # xmin = row['xmin'] / width 
        #xmax = row['xmax'] / width
        #ymin = row['ymin'] / height
        #ymax = row['ymax'] / height
        center_x.append(row['x'] / width)
        center_y.append(row['y'] / height)

        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        # 'image/height': dataset_util.int64_feature(height),
        # 'image/width': dataset_util.int64_feature(width),
        # 'image/filename': dataset_util.bytes_feature(filename),
        # 'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_jpg),
        # 'image/format': dataset_util.bytes_feature(image_format),
        # 'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        # 'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        # 'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        # 'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/bbox/center_x': _float_list_feature(center_x),
        'image/object/bbox/center_y': _float_list_feature(center_y),
        # 'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        # 'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == '__main__':
    tf.app.run()
