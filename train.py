import tensorflow as tf 
import numpy as np
from net.efficentnet import EfficentNet

img = np.random.random((512, 512, 3))
img = (img*255).astype('uint8')
img = np.expand_dims(img, axis=0)
print(img.shape)
img = tf.convert_to_tensor(img, dtype='float32')

EfficentNet = EfficentNet()
# init = tf.global_variables_initializer()
out = EfficentNet.net(img=img)
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(out)
    
print('Done!')