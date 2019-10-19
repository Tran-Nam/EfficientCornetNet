import tensorflow as tf  
from .model import Model

class EfficentNet():
    def __init__(self):
        self.model = Model()
    def net(self, img, scope='EfficentNet'):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(img, 32, kernel_size=3, strides=(2, 2), padding='same')
            x = tf.layers.conv2d(x, 32, kernel_size=3, strides=(2, 2), padding='same')
            x = self.model.MBConv6(x, 16, expands=1, scope='MBConv1_0')
            x = self.model.MBConv6(x, 24, kernel_size=3, scope='MBConv6_3x3_0')
            x = self.model.MBConv6(x, 24, kernel_size=3, scope='MBConv6_3x3_1')
            x = self.model.MBConv6(x, 40, kernel_size=5, scope='MBConv6_5x5_2')
            x = self.model.MBConv6(x, 40, kernel_size=5, scope='MBConv6_5x5_3')
            x = self.model.MBConv6(x, 80, kernel_size=3, scope='MBConv6_3x3_4')
            x = self.model.MBConv6(x, 80, kernel_size=3, scope='MBConv6_3x3_5')
            x = self.model.MBConv6(x, 80, kernel_size=3, scope='MBConv6_3x3_6')
            x = self.model.MBConv6(x, 112, kernel_size=5, scope='MBConv6_5x5_7')
            x = self.model.MBConv6(x, 112, kernel_size=5, scope='MBConv6_5x5_8')
            x = self.model.MBConv6(x, 112, kernel_size=5, scope='MBConv6_5x5_9')
            x = self.model.MBConv6(x, 192, kernel_size=5, scope='MBConv6_5x5_10')
            x = self.model.MBConv6(x, 192, kernel_size=5, scope='MBConv6_5x5_11')
            x = self.model.MBConv6(x, 192, kernel_size=5, scope='MBConv6_5x5_12')
            x = self.model.MBConv6(x, 192, kernel_size=5, scope='MBConv6_5x5_13')
            x = self.model.MBConv6(x, 320, kernel_size=3, scope='MBConv6_3x3_14')
            # x = tf.layers.conv2d(x, 1280)

            return x