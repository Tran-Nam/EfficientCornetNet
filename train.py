import tensorflow as tf 
import numpy as np
import os

from net.efficentnet import EfficentNet
from helper.generate_gt import gen_gt
from module.loss import offset_loss, focal_loss
# from processing.dataloader import DataLoader
from processing.dataloader_2 import input_fn
import config

_img = np.random.random((1, 512, 512, 3))
_heatmap = np.random.random((1, 128, 128, 4))
_offset = np.random.random((1, 128, 128, 2))
_mask = np.zeros((1, 128, 128))
_mask[:, 20:40, 50:60] = 1

class Train():
    def __init__(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # self.batchsize = config.BATCH_SIZE
        # self.dataloader = DataLoader(image_dir='../../images')
        # self.dataloader.create_pairs()
        # self.it = iter(self.dataloader.train_ds)
        
        # self.lr = config.LEARNING_RATE
        # self.decay_step = config.DECAY_STEP
        # self.dacay_rate = config.DECAY_RATE
        self.num_steps = 10

        # self.pretrained = config.PRETRAINED
        # self.model_path = config.MODEL_PATH
        # self.save_model_dir = config.MODEL_DIR
        self.net = EfficentNet()
        self.focal_loss = focal_loss
        self.offset_loss = offset_loss

        # self.dataset = input_fn()
        # self.iterator = self.dataset.make_initializable_iterator()
        # self.iterator_init_op = self.iterator.initializer

        # print('Done!')
    
    def load_ckpt(self, saver, sess, model_dir):
        """
        load pretrained weight
        :param saver: saver object
        :param sess: tf session
        :param model_dir: path to checkpoint file
        """
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(model_dir, ckpt_name))
            print('Restore model from {}'.format(ckpt_name))
            return True 
        else:
            return False

    # def feed_dict(self):
    #     # image_batch, heatmap_batch, offset_batch, mask_batch = self.dataloader.next_batch()
    #     # image_batch, heatmap_batch, offset_batch, mask_batch = next(self.it)
    #     _img = np.random.random((10, 512, 512, 3))
    #     _heatmap = np.random.random((10, 128, 128, 4))
    #     _offset = np.random.random((10, 128, 128, 2))
    #     _mask = np.zeros((10, 128, 128))
    #     _mask[:, 20:40, 50:60] = 1
    #     return {'image': _img, \
    #         'heatmap': _heatmap, \
    #         'offset': _offset, \
    #         'mask': _mask}

    def train(self):
        # image_batch, heatmap_batch, offset_batch = self.dataloader.next_batch() # load data
        # output = self.net.net(image_batch)
        # heatmap, offset = output
        # steps = tf.Variable(0, name='global_step', trainable=False)
        # lr = tf.train.exponential_decay(self.lr, steps, self.decay_step, self.decay_rate, staircase=True, name='learning_rate')
        # optim = tf.train.AdamOptimizer(learning_rate=1e-3)
        
        # placeholder
        image = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='image')
        heatmap = tf.placeholder(tf.float32, shape=[None, 128, 128, 4], name='heatmap')
        offset = tf.placeholder(tf.float32, shape=[None, 128, 128, 2], name='offset')
        mask = tf.placeholder(tf.float32, shape=[None, 128, 128], name='mask')
        gt = (heatmap, offset, mask)

        # batch, init_op = self.iterator.get_next()
        batch, iterator_init_op = input_fn()
        image = batch['image']
        heatmap = batch['heatmap']
        offset = batch['offset']
        mask = batch['mask']
        print(image.shape)

        # print('Placeholder')
        # def feed_dict():
        #     # image_batch, heatmap_batch, offset_batch, mask_batch = self.dataloader.next_batch()
        #     # image_batch, heatmap_batch, offset_batch, mask_batch = next(self.it)
        #     _img = np.random.random((10, 512, 512, 3))
        #     _heatmap = np.random.random((10, 128, 128, 4))
        #     _offset = np.random.random((10, 128, 128, 2))
        #     _mask = np.zeros((10, 128, 128))
        #     _mask[:, 20:40, 50:60] = 1
        #     return {image: _img, \
        #         heatmap: _heatmap, \
        #         offset: _offset, \
        #         mask: _mask}
        
        # build graph
        # with tf.variable_scope('', reuse=True):
        output = self.net.net(img=image)
        heatmap_det, offset_det = output 
        # loss = self.net.loss(output, gt)
        f_loss = self.focal_loss(heatmap_det, heatmap)
        o_loss = self.offset_loss(offset_det, offset, mask)
        loss = tf.add(f_loss, o_loss)
        # print(loss)
        # print(f_loss)
        # print(o_loss)
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        # print(train_op)
        # writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
            # trainable_variables = tf.trainable_variables()
        
        init = tf.global_variables_initializer() 
        
        # update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update):
        
        
        # saver = tf.train.Saver(max_to_keep=10)
        # config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        # config_gpu.gpu_options.allow_growth = True 
        # sess = tf.Session(config=config_gpu)
        sess = tf.Session()
        print('Create session')
        sess.run(init)
        # input()
        sess.run(iterator_init_op)
        # input()
        # sess.run(self.iterator_init_op)
        # sess.run(batch.initialize())

        # print(self.num_steps)
        # epoch = 0
        # if self.pretrained:
        #     if self.load_ckpt(saver, sess, self.save_model_dir):
        #         print('[*] Load SUCCESS!')
        #     else:
        #         print('[*] Load FAIL ...')
        
        for step in range(self.num_steps):
            # print(step)
            f, o, loss_ = sess.run([f_loss, o_loss, loss])
            print('step %d, loss %g, focal loss %g, offset loss %g'%(step, loss_, f, o))
            # print(loss_)
            # if step%self.interval_save==0 and step>0:
            #     saver.save(sess, self.model_path, epoch)
            #     epoch += 1








if __name__=='__main__':
    t = Train()
    t.train()


