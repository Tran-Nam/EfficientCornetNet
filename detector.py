import tensorflow as tf 
import cv2 
from PIL import Image
import numpy as np 
from helper.utils import decodeDets, resize_image


class ObjectDetection():
    def __init__(self):
        import config
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(config.FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
        self.input_node_name = 'data_pipeline/IteratorGetNext'
        self.heatmap_node_name = 'EfficentCornerNet/heat/conv2d/BiasAdd'
        self.offset_node_name = 'EfficentCornerNet/offset/conv2d/BiasAdd'

        self.input_node = self.detection_graph.get_tensor_by_name(self.input_node_name + ':1') # image
        self.heatmap_node = self.detection_graph.get_tensor_by_name(self.heatmap_node_name + ':0')
        self.offset_node = self.detection_graph.get_tensor_by_name(self.offset_node_name + ':0')

    def detect(self, image):
        self.image = image
        im_h, im_w = self.image.shape[:2]
        image_in, new_size = resize_image(self.image)
        self.im_h_resize, self.im_w_resize = new_size
        self.ratio = 512 / max(im_h, im_w)
        image_expand = np.expand_dims(image_in, axis=0) #1x512x512x3

        with self.detection_graph.as_default():
            with tf.Session() as sess:
                self.heatmap, self.offset = sess.run([self.heatmap_node, self.offset_node], feed_dict={self.input_node: image_expand})
        self.heatmap = np.squeeze(self.heatmap, axis=0)
        self.heatmap = self.sigmoid(self.heatmap) # convert to 0-1
        self.offset = np.squeeze(self.offset, axis=0)
        return self 
    
    def visual(self):
        heatmap = np.max(self.heatmap, axis=2) # 128x128
        heatmap = (heatmap*255/np.max(heatmap)).astype('uint8')
        heatmap = cv2.resize(heatmap, None, fx=4, fy=4)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        padding_x = (512 - self.im_w_resize)//2
        padding_y = (512 - self.im_h_resize)//2
        padding = np.array([padding_y, padding_x]) # correspond 512x512 image

        heatmap = heatmap[padding_y: padding_y+self.im_h_resize, 
                        padding_x: padding_x+self.im_w_resize, :]
        heatmap_origin = cv2.resize(heatmap, None, fx=1/self.ratio, fy=1/self.ratio)

        corners = decodeDets(self.heatmap, self.offset) # correspond 512x512 image
        corners = np.maximum(0, corners - padding)
        corners_origin = (corners / self.ratio).astype('int32')

        im_visual = cv2.addWeighted(self.image, 0.3, heatmap_origin, 0.7, 0)
        cv2.polylines(im_visual, [corners_origin[:, ::-1]], True, (0, 255, 0), 2) # reverse x, y to visual
        cv2.polylines(heatmap, [corners[:, ::-1]], True, (0, 255, 0), 2)
        # cv2.cvtColor(im_visual, cv2.COLOR_RGB2BGR) 
        # cv2.imwrite('b.png', im_visual)
        # cv2.imwrite('c.png', heatmap)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
