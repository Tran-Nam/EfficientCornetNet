import tensorflow as tf 
import numpy as np
import os

from net.efficentnet import EfficentNet
from helper.generate_gt import gen_gt
from module.loss import offset_loss, focal_loss
from processing.dataloader import DataLoader
import config

class Train():
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.batchsize = config.BATCH_SIZE
        self.dataloader = DataLoader(image_dir='../../images')
        self.dataloader.create_pairs()
        self.it = iter(self.dataloader.train_ds)
        
        self.lr = config.LEARNING_RATE
        self.decay_step = config.DECAY_STEP
        self.dacay_rate = config.DECAY_RATE
        self.num_steps = 1000

        self.pretrained = config.PRETRAINED
        self.model_path = config.MODEL_PATH
        self.save_model_dir = config.MODEL_DIR
        self.net = EfficentNet()
    
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

    def feed_dict(self):
        # image_batch, heatmap_batch, offset_batch, mask_batch = self.dataloader.next_batch()
        image_batch, heatmap_batch, offset_batch, mask_batch = next(self.it)
        return {image: image_batch, \
            heatmap: heatmap_batch, \
            offset: offset_batch, \
            mask: mask_batch}

    def train(self):
        # image_batch, heatmap_batch, offset_batch = self.dataloader.next_batch() # load data
        # output = self.net.net(image_batch)
        # heatmap, offset = output
        steps = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(self.lr, steps, self.decay_step, self.decay_rate, staircase=True, name='learning_rate')
        optim = tf.train.AdamOptimizer(learning_rate=lr)
        
        # placeholder
        image = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='image')
        heatmap = tf.placeholder(tf.float32, shape=[None, 128, 128, 4], name='heatmap')
        offset = tf.placeholder(tf.float32, shape=[None, 128, 128, 2], name='offset')
        mask = tf.placeholder(tf.float32, shape=[None, 128, 128], name='mask')
        gt = (heatmap, offset, mask)
        
        # build graph
        with tf.variable_scope('', reuse=True):
            output = self.net.net(img=image)
            heatmap_det, offset_det = output 
            loss = self.net.loss(output, gt)
            trainable_variables = tf.trainable_variables()
        
        init = tf.global_variables_initializer() 
        
        update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update):
            train_op = optim.minimize(loss, var_list=trainable_variables)
        
        saver = tf.train.Saver(max_to_keep=10)
        config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config_gpu.gpu_options.allow_growth = True 
        sess = tf.Session(config=config_gpu)
        sess.run(init)

        print(self.num_steps)
        epoch = 0
        if self.pretrained:
            if self.load_ckpt(saver, sess, self.save_model_dir):
                print('[*] Load SUCCESS!')
            else:
                print('[*] Load FAIL ...')
        
        for step in range(self.num_steps):
            _, loss_, lr_ = sess.run([train_op, loss, lr], feed_dict=self.feed_dict())
            print('step %d, loss %g, lr %g'%(step, loss_, lr_))

            # if step%self.interval_save==0 and step>0:
            #     saver.save(sess, self.model_path, epoch)
            #     epoch += 1








if __name__=='__main__':
    t = Train()
    t.train()


