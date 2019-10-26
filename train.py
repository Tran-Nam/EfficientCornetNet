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
        self.dataloader = DataLoader()
        self.dataloader.create_pairs()
        self.pretrained = config.PRETRAINED
        self.model_path = config.MODEL_PATH
        self.net = EfficentNet()
    
    def load_ckpt(self, saver, sess, model_path):
        """
        load pretrained weight
        :param saver: saver object
        :param sess: tf session
        :param model_path: path to checkpoint file
        """
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(model_path, ckpt_name))
            print('Restore model from {}'.format(ckpt_name))
            return True 
        else:
            return False


    def train(self):
        image_batch, heatmap_batch, offset_batch = self.dataloader.next_batch() # load data
        output = self.net.net(image_batch)
        heatmap, offset = output
        










