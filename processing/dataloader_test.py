import tensorflow as tf 
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
tf.enable_eager_execution()

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
# STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

data_dir = '../data'

data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*')))
print(image_count)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes='a')

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

for f in list_ds.take(5):
    print(f.numpy())

def get_label(file_path):
    # convert the path to a list of path components
    # parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    return 'a'

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize_images(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

labeled_ds = list_ds.map(process_path, num_parallel_calls=6)

for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print(image.shape)
    cv2.imwrite('a.png', (image.numpy()*255).astype('uint8'))
    plt.imsave('b.png', (image.numpy()*255).astype('uint8'))
    print(type(image))
    print("Label: ", label.numpy())
