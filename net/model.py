import tensorflow as tf 

class Model():

    def conv_block(self, inputs, out_dim, kernel_size=3, strides=(1, 1), \
        use_batchnorm=True, use_relu=True, is_training=True, scope='conv_block'): # conv block conv-bn-re
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(inputs, out_dim, kernel_size, strides, padding='same')
            if use_batchnorm:
                x = tf.contrib.layers.batch_norm(x, is_training=is_training)
            if use_relu:
                x = tf.nn.relu(x)
            return x 
    
    def res_unit(self, inputs, out_dim, kernel_size=3, strides=(1, 1), \
        is_training=True, scope='res_unit'):
        in_dim = inputs.get_shape().as_list()[3] #inputs: [b, h, w, c]
        with tf.variable_scope(scope):
            x = self.conv_block(inputs, out_dim=out_dim, kernel_size=kernel_size, \
                strides=strides, is_training=is_training, scope='conv_block1')
            x = self.conv_block(x, out_dim=out_dim, kernel_size=kernel_size, \
                strides=strides, use_relu=False, is_training=is_training, scope='conv_block2')
            skip = tf.layers.conv2d(inputs, out_dim, kernel_size, strides, padding='same')
            res_unit = tf.add(skip, x)
            res_unit = tf.nn.relu(res_unit)
            return res_unit

    # def res_block(self, inputs, out_dim, n_unit, kernel_size=3, strides=(1, 1), \
    #     is_training=True, scope='res_block'):
    #     assert n_unit > 0
    #     with tf.variable_scope(scope):
    #         x = self.res_unit(inputs, out_dim=out_dim, kernel_size=kernel_size, \
    #             strides=strides, is_training=is_training, scope='res_block0')
    #         for i in range(1, n_unit):
    #             x = self.res_unit(x, out_dim=out_dim, kernel_size=kernel_size, \
    #                 strides=strides, is_training=is_training, scope='res_block{}'.format(i)))
    #         return x 

    def DWConv_block(self, inputs, out_dim, kernel_size=3, use_batchnorm=True, use_relu=True, \
        is_training=True, scope='DWConv_block'):
        with tf.variable_scope(scope):
            x = tf.contrib.layers.separable_conv2d(inputs, out_dim, kernel_size, depth_multiplier=1, activation_fn=None) # not use relu here
            if use_batchnorm:
                x = tf.contrib.layers.batch_norm(x, is_training=is_training)
            if use_relu:
                x = tf.nn.relu(x)
            return x

    def MBConv6(self, inputs, out_dim, kernel_size=3, expands=6, is_training=True, scope='MBConv6'):
        print(inputs.get_shape().as_list())
        in_dim = inputs.get_shape().as_list()[3]
        expands_dim = expands * out_dim
        with tf.variable_scope(scope):
            base = tf.layers.conv2d(inputs, out_dim, kernel_size=1, padding='same')
            x = self.conv_block(base, expands_dim, kernel_size=1, scope='conv1x1_bn_relu')
            x = self.DWConv_block(x, expands_dim, kernel_size=3, scope='DwConv_bn_relu')
            x = self.conv_block(x, out_dim, kernel_size=1, use_relu=False, scope='conv1x1_bn')
            x = tf.add(base, x)
            x = tf.nn.relu(x)
            return x
    
            