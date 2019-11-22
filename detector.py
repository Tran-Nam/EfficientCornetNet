import tensorflow as tf 
import cv2
from PIL import Image
import numpy as np

model_path = './model/model.pb'

def sigmoid(x):
    return 1/(1+np.exp(-x))

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        # print(od_graph_def)
        tf.import_graph_def(od_graph_def, name='')
    
# nodes = [n.name + '=>' + n.op for n in od_graph_def.node if n.op in ('Placeholder', 'BiasAdd')]
# nodes_ = [n.name + ':' + n.op for n in od_graph_def.node]
# print(nodes)
# for node in nodes_:
#     print(node)


im = cv2.imread('b.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (512, 512))
im = np.expand_dims(im, axis=0)
# im = np.zeros((1, 128, 128, 4))


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        heatmap_node = 'EfficentCornerNet/heat/conv2d/BiasAdd:0'
        offset_node = 'EfficentCornerNet/offset/conv2d/BiasAdd:0'
        # input_node_name = 'data_pipeline/IteratorGetNext'
        # input_node = detection_graph.get_tensor_by_name(input_node_name+':0')
        # input_tensor = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='data_pipeline/IteratorGetNext')
        # heat, off = tf.import_graph_def(od_graph_def, input_map={'data_pipeline/IteratorGetNext': input_tensor}, \
        #     return_elements=[heatmap_node, offset_node])
        # print('BEFORE IMPORT GRAPH DEF' + '*'*50 + '\n', heatmap_node, offset_node)
        # _ = tf.import_graph_def(od_graph_def, name='')
        # print('AFTER IMPORT' + '*'*50)

        # nodes = [n.name for n in detection_graph.get_operations()]
        # print(nodes)

        input_node_name = 'data_pipeline/IteratorGetNext'
        input_node = detection_graph.get_tensor_by_name(input_node_name + ':1')
        heatmap = detection_graph.get_tensor_by_name('EfficentCornerNet/heat/conv2d/BiasAdd:0')
        offset = detection_graph.get_tensor_by_name('EfficentCornerNet/offset/conv2d/BiasAdd:0')

        heatmap, offset = sess.run([heatmap, offset], feed_dict={input_node: im})
        heatmap = sigmoid(heatmap)

        print(heatmap.shape, offset.shape)
        
        heatmap = np.squeeze(heatmap, axis=0)
        offset = np.squeeze(offset, axis=0)

        corner = np.max(heatmap, axis=2)
        # corner = heatmap[:, :, 0]
        corner = (corner*255/np.max(corner)).astype('uint8')

        # cv2.imshow('a', cv2.applyColorMap(corner, cv2.COLORMAP_JET))
        print(corner.shape)
        cv2.imshow('a', corner)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

