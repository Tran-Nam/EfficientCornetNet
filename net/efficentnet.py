import tensorflow as tf  
import sys
sys.path.append('..')
from .model import Model
from module.loss import focal_loss, offset_loss, focal_loss_2, loss

class EfficentNet():
    def __init__(self):
        self.model = Model()
        self.focal_loss = focal_loss
        self.offset_loss = offset_loss
        self.loss = loss
        self.gamma = 1
    def net(self, img, scope='EfficentCornerNet'):
        with tf.variable_scope(scope):
            # conv1 = tf.layers.conv2d(img, 32, kernel_size=3, strides=(1, 1), padding='same')
            conv1 = self.model.conv_block(img, 32, kernel_size=3)
            pool1 = tf.layers.max_pooling2d(conv1, 2, strides=(2, 2), padding='same') #256x256x32
            # x = tf.layers.conv2d(x, 32, kernel_size=3, strides=(1, 1), padding='same')
            # x = tf.layers.max_pooling2d(x, 2, strides=(2, 2), padding='same')

            # Efficent Net
            MBConv1_0 = self.model.MBConv6(pool1, 16, expands=1, scope='MBConv1_0') #256x256x16
            # MBConv6_0 = self.model.MBConv6(MBConv1_0, 24, kernel_size=3, max_pool=True, scope='MBConv6_3x3_0') #128x128x24
            MBConv6_0 = self.model.MBConv6(MBConv1_0, 24, kernel_size=3, strides=(2, 2), scope='MBConv6_3x3_0') #128x128x24
            MBConv6_1 = self.model.MBConv6(MBConv6_0, 24, kernel_size=3, scope='MBConv6_3x3_1')
            # MBConv6_2 = self.model.MBConv6(MBConv6_1, 40, kernel_size=5, max_pool=True, scope='MBConv6_5x5_2') #64x64x40
            MBConv6_2 = self.model.MBConv6(MBConv6_1, 40, kernel_size=5, strides=(2, 2), scope='MBConv6_5x5_2') #64x64x40
            MBConv6_3 = self.model.MBConv6(MBConv6_2, 40, kernel_size=5, scope='MBConv6_5x5_3')
            MBConv6_4 = self.model.MBConv6(MBConv6_3, 80, kernel_size=3, scope='MBConv6_3x3_4')
            MBConv6_5 = self.model.MBConv6(MBConv6_4, 80, kernel_size=3, scope='MBConv6_3x3_5')
            MBConv6_6 = self.model.MBConv6(MBConv6_5, 80, kernel_size=3, scope='MBConv6_3x3_6')
            # MBConv6_7 = self.model.MBConv6(MBConv6_6, 112, kernel_size=5, max_pool=True, scope='MBConv6_5x5_7') #32x32x112
            MBConv6_7 = self.model.MBConv6(MBConv6_6, 112, kernel_size=5, strides=(2, 2), scope='MBConv6_5x5_7') #32x32x112
            MBConv6_8 = self.model.MBConv6(MBConv6_7, 112, kernel_size=5, scope='MBConv6_5x5_8')
            MBConv6_9 = self.model.MBConv6(MBConv6_8, 112, kernel_size=5, scope='MBConv6_5x5_9')
            # MBConv6_10 = self.model.MBConv6(MBConv6_9, 192, kernel_size=5, max_pool=True, scope='MBConv6_5x5_10') #16x16x192
            MBConv6_10 = self.model.MBConv6(MBConv6_9, 192, kernel_size=5, strides=(2, 2), scope='MBConv6_5x5_10') #16x16x192
            MBConv6_11 = self.model.MBConv6(MBConv6_10, 192, kernel_size=5, scope='MBConv6_5x5_11')
            MBConv6_12 = self.model.MBConv6(MBConv6_11, 192, kernel_size=5, scope='MBConv6_5x5_12')
            MBConv6_13 = self.model.MBConv6(MBConv6_12, 192, kernel_size=5, scope='MBConv6_5x5_13')
            MBConv6_14 = self.model.MBConv6(MBConv6_13, 320, kernel_size=3, scope='MBConv6_3x3_14') #16x16x320

            # Upsampling
            UPConv_0 = self.model.UPConv(MBConv6_14, MBConv6_9, kernel_size=3, scope='UpConv_0') #32x32x112
            UPConv_1 = self.model.UPConv(UPConv_0, MBConv6_6, kernel_size=3, scope='UpConv_1') #64x64x40
            UPConv_2 = self.model.UPConv(UPConv_1, MBConv6_1, kernel_size=3, scope='UpConv_2') #128x128x24
            
            # get output
            heat = self.model.heat(UPConv_2)
            offset = self.model.offset(UPConv_2)

            return heat, offset


    def loss(self, output, gt, scope='loss'):
        heatmap, offset = output 
        heatmap_gt, offset_gt, mask = gt 
        with tf.variable_scope(scope):
            # with tf.variable_scope('focal_loss'):
            #     f_loss = self.focal_loss(heatmap, heatmap_gt)
            # with tf.variable_scope('offset_loss'):
            #     o_loss = self.offset_loss(offset, offset_gt, mask)
            f_loss = self.focal_loss(heatmap, heatmap_gt)
            o_loss = self.offset_loss(offset, offset_gt, mask)
            return tf.add(f_loss, o_loss)
            # heatmap, offset = output 
            # heatmap_gt, offset_gt, mask = gt 
            # f_loss = self.focal_loss(heatmap, heatmap_gt)
            # o_loss = self.offset_loss(offset, offset_gt, mask)
            # return self.focal_loss(heatmap, heatmap_gt) + self.offset_loss(offset, offset_gt, mask)
            # return self.loss(output, gt)
            # return f_loss + self.gamma*o_loss
            # return loss
