import tensorflow as tf 
import numpy as np
from net.efficentnet import EfficentNet

_img = np.random.random((512, 512, 3))
_img = (_img*255).astype('uint8')
_img = np.expand_dims(_img, axis=0)
print(_img.shape)
# _img = tf.convert_to_tensor(_img, dtype='float32')
img = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])

EfficentNet = EfficentNet()
# init = tf.global_variables_initializer()
out = EfficentNet.net(img=img)
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    heatmap, offset = sess.run(out, feed_dict={img: _img})
    # print(offset.shape) ### numpy array !!!!!!!!!!! 0_0
    print(type(offset))
    print(type(heatmap))
    print(offset.shape, heatmap.shape)
    
    
print('Done!')