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
        
        self.lr = config.LEARNING_RATE
        self.decay_step = config.DECAY_STEP
        self.decay_rate = config.DECAY_RATE
        self.num_steps = 1000
        self.model_dir = config.MODEL_DIR

        self.pretrained = config.PRETRAINED
        self.model_path = config.MODEL_PATH
        # self.save_model_dir = config.MODEL_DIR
        self.net = EfficentNet()
        self.focal_loss = focal_loss
        self.offset_loss = offset_loss

        # self.dataset = input_fn()
        # self.iterator = self.dataset.make_initializable_iterator()
        # self.iterator_init_op = self.iterator.initializer

        # print('Done!')
    
    def load_ckpt(self, saver, sess, model_path):
        """
        load pretrained weight
        :param saver: saver object
        :param sess: tf session
        :param model_dir: path to checkpoint file
        """
        # ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(model_path)
        if os.path.exists(model_path + '.meta'):
            # ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # print(model_path)
            saver.restore(sess, model_path)
            print('Restore model from {}'.format(model_path))
            return True 
        else:
            return False


    def train(self):
        # image_batch, heatmap_batch, offset_batch = self.dataloader.next_batch() # load data
        # output = self.net.net(image_batch)
        # heatmap, offset = output
        # steps = tf.Variable(0, name='global_step', trainable=False)
        # lr = tf.train.exponential_decay(self.lr, steps, self.decay_step, self.decay_rate, staircase=True, name='learning_rate')
        # optim = tf.train.AdamOptimizer(learning_rate=1e-3)
        
        # placeholder
        # image = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='image')
        # heatmap = tf.placeholder(tf.float32, shape=[None, 128, 128, 4], name='heatmap')
        # offset = tf.placeholder(tf.float32, shape=[None, 128, 128, 2], name='offset')
        # mask = tf.placeholder(tf.float32, shape=[None, 128, 128], name='mask')
        # gt = (heatmap, offset, mask)

        # batch, init_op = self.iterator.get_next()
        with tf.variable_scope('data_pipeline'):
            batch, iterator_init_op = input_fn()
        image = batch['image']
        heatmap = batch['heatmap']
        offset = batch['offset']
        mask = batch['mask']
        gt = (heatmap, offset, mask)
        # print(image.shape)

        # build graph
        # with tf.variable_scope('', reuse=True):
        output = self.net.net(img=image)
        heatmap_det, offset_det = output
        # loss = self.net.loss(output, gt)
        # loss = self.net.loss(output, gt)
        with tf.variable_scope('loss'):
            f_loss = self.focal_loss(heatmap_det, heatmap)
            f_loss_check = tf.debugging.check_numerics(f_loss, "Focal loss is NaN", name="DEBUG")
            o_loss = self.offset_loss(offset_det, offset, mask)
            o_loss_check = tf.debugging.check_numerics(o_loss, "Focal loss is NaN", name="DEBUG")
            loss = tf.add(f_loss, o_loss)
            loss = tf.cast(loss, dtype=tf.float64)
        # print(loss)
        # print(f_loss)
        # print(o_loss)
        with tf.variable_scope('optimizer'):
            # with tf.control_dependencies([f_loss_check, o_loss_check]):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # increment_global_step = tf.assign(global_step, global_step + 1)
            lr=tf.train.exponential_decay(self.lr,global_step,self.decay_step,self.decay_rate,staircase=True, name= 'learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss, global_step)
        # print(train_op)
        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
            # trainable_variables = tf.trainable_variables()
        saver = tf.train.Saver(max_to_keep=3)
        init = tf.global_variables_initializer() 

        # f_summary = tf.summary.scalar(name='loss', tensor=loss)
        tf.summary.scalar('focal_loss', f_loss)
        tf.summary.scalar('offset_loss', o_loss)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('lr', lr)
        tf.summary.image('input', image, max_outputs=5)
        tf.summary.image('top_left_gt', heatmap[:, :, :, 0:1], max_outputs=5)
        tf.summary.image('top_left', heatmap_det[:, :, :, 0:1], max_outputs=5)
        merge = tf.summary.merge_all()
        
        # update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update):
        
        
        # saver = tf.train.Saver(max_to_keep=10)
        # config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        # config_gpu.gpu_options.allow_growth = True 
        # sess = tf.Session(config=config_gpu)
        sess = tf.Session()
        # print('Create session')
        sess.run(init)
        # input()
        sess.run(iterator_init_op)
        # input()
        # sess.run(self.iterator_init_op)
        # sess.run(batch.initialize())

        # print(self.num_steps)
        # epoch = 0
        if self.pretrained:
            if self.load_ckpt(saver, sess, self.model_path):
                print('[*] Load SUCCESS!')
            else:
                print('[*] Load FAIL ...')
        
        for step in range(self.num_steps):
            # print(step)
            # sess.run(debug)
            loss_, f_loss_, o_loss_, _ = sess.run([loss, f_loss, o_loss, train_op])
            print('step %d, loss %g, focal %g, offset %g'%(step, loss_, f_loss_, o_loss_))

            if step%config.INTERVAL_SAVE==0 and step > 0:
                summary = sess.run(merge)
                writer.add_summary(summary, step)
                saver.save(sess, os.path.join(self.model_dir, str(step) + '.ckpt'))
            # print(loss_)
            # if step%self.interval_save==0 and step>0:
            #     saver.save(sess, self.model_path, epoch)
            #     epoch += 1








if __name__=='__main__':
    t = Train()
    t.train()


