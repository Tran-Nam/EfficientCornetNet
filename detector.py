import tensorflow as tf 
import cv2
from PIL import Image
import numpy as np

model_path = './model/model.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
# nodes = [n.name for n in detection_graph.get_operations()]
# print(nodes)


im = cv2.imread('a.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (512, 512))
im = np.expand_dims(im, axis=0)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # data_pipeline/IteratorGetNext
        heatmap = detection_graph.get_tensor_by_name('EfficentCornerNet/heat/conv2d/BiasAdd:0')
        offset = detection_graph.get_tensor_by_name('EfficentCornerNet/offset/conv2d/BiasAdd:0')

        heatmap, offset = sess.run([heatmap, offset])

