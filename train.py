import tensorflow as tf 
import numpy as np
from net.efficentnet import EfficentNet
from helper.generate_gt import gen_gt
from module.loss import offset_loss, focal_loss

_img = np.random.random((10, 512, 512, 3))
_heatmap = np.random.random((10, 128, 128, 4))
_offset = np.random.random((10, 128, 128, 2))
_mask = np.zeros((10, 128, 128))
_mask[:, 20:40, 50:60] = 1
_img = (_img*255).astype('uint8')
# _img = np.expand_dims(_img, axis=0)
# _heatmap = np.expand_dims(_heatmap, axis=0)
# _offset = np.expand_dims(_offset, axis=0)
# _mask = np.expand_dims(_mask, axis=0)

print(_img.shape)
# _img = tf.convert_to_tensor(_img, dtype='float32')
img = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
heatmap_gt = tf.placeholder(tf.float32, shape=[None, 128, 128, 4])
offset_gt = tf.placeholder(tf.float32, shape=[None, 128, 128, 2])
mask = tf.placeholder(tf.float32, shape=[None, 128, 128])

EfficentNet = EfficentNet()
# init = tf.global_variables_initializer()
output = EfficentNet.net(img=img)
heatmap, offset = output
gt = (heatmap_gt, offset_gt, mask)
loss = EfficentNet.loss(output, gt)
# f_loss = focal_loss(heatmap, heatmap_gt)
# o_loss = offset_loss(offset, offset_gt, mask)
# total_loss = f_loss + o_loss
# focal_loss = loss.focal_loss2(heatmap, img)
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    heatmap, offset, loss = sess.run([heatmap, offset, loss], feed_dict={img: _img, heatmap_gt: _heatmap, offset_gt: _offset, mask: _mask})
    
    # print(offset.shape) ### numpy array !!!!!!!!!!! 0_0
    print(type(offset))
    print(type(heatmap))
    # print(offset.shape, heatmap.shape)
    # print(loss.eval())
    print(loss)
    
    
print('Done!')