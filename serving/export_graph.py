import os 
import argparse 
import tensorflow as tf 
from tensorflow.summary import FileWriter
from tensorflow.python.tools import freeze_graph
# import tensorflow.data

def save_pbtxt(model_name):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.import_meta_graph(model_name+'.meta', clear_devices=True)
            tf.train.write_graph(sess.graph.as_graph_def(), '../model', 'ECN.pbtxt', as_text=True)

"""
def freeze_graph(model_name, output_node_names, output_dir):
    # _ = tf.data
    # _ = tf.contrib.resampler

    with tf.Session(graph=tf.Graph()) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        output_graph = os.path.join(output_dir, 'frozen_model.pb')
        # print('Before restore ' + '>'*20)
        saver = tf.train.import_meta_graph(model_name+'.meta', clear_devices=True)
        # graph_def = tf.get_default_graph().as_graph_def()
        # node_list=[n.name for n in graph_def.node if 'optimizer' not in n.name]
        # print(node_list)
        # FileWriter("__tb", sess.graph)
        # print('After restore ' + '>'*20)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, 
            tf.get_default_graph().as_graph_def(),
            output_node_names.split(',')
        )

        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        print('%d ops in the final graph', len(output_graph_def.node))

    return output_graph_def
"""

def freeze_graph_2():
    freeze_graph.freeze_graph('../model/ECN.pbtxt', 
                            '',
                            False,
                            input_checkpoint='../checkpoint/model-29001',
                            output_node_names='EfficentCornerNet/heat/conv2d/BiasAdd,EfficentCornerNet/offset/conv2d/BiasAdd',
                            output_graph='../model/model.pb',
                            clear_devices=True,
                            restore_op_name = "save/restore_all",
                            filename_tensor_name = "save/Const:0",
                            initializer_nodes='')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='../checkpoint/model-29001', help='Checkpoint name')
    parser.add_argument('--output_node_names', type=str, default='../EfficentCornerNet/heat/conv2d/BiasAdd', help='Output node')
    parser.add_argument('--output_dir', type=str, default='../checkpoint', help='Output dir to place frozen model')
    args = parser.parse_args()

    # freeze_graph(args.model_name, args.output_node_names, args.output_dir)
    # save_pbtxt(args.model_name)
    freeze_graph_2()
