BATCH_SIZE = 4
PRETRAINED = False 
MODEL_PATH = './checkpoint/10.ckpt'

DATA_PATH = '../data/data_2.tfrecords'

LEARNING_RATE = 1e-5
DECAY_STEP = 1000
DECAY_RATE = 0.95 

INTERVAL_SAVE = 10
MODEL_DIR = 'checkpoint'

FROZEN_GRAPH = './model/model.pb'