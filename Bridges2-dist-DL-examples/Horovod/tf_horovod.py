import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd
from tensorflow.python.keras import optimizers
from timeit import default_timer as timer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tqdm import tqdm
import numpy as np
import time

device = 'GPU'
# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.list_physical_devices('GPU')
print('number of GPUs:',len(gpus))
print(gpus,hvd.size())
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = argparse.ArgumentParser(description='Input values.')
parser.add_argument('-bz', type=int, default=128, help='Batch size per replica')
parser.add_argument('-image_size', type=int, default=128, help='Image size (number of pixels per size)')
parser.add_argument('-epoch_num', type=int, default=5, help='Number of training epochs')
parser.add_argument('-mp', action='store_true', help='Mixed precision training')
parser.add_argument('-imagenet', action='store_true', help='Using Imagenet dataset')

args = parser.parse_args()
batch_size = args.bz
image_size = args.image_size
mixed_precision = args.mp
using_imagenet = args.imagenet
epoch_num = args.epoch_num
if hvd.local_rank() == 0:
    print('Batch size=',batch_size)
    print('Image size=',image_size)
    print('Mixed precision training=',mixed_precision)
    print('Number of GPUs=',hvd.size())

# Enable mixed precision training if assigned.
if  mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

# Reading in imagenet dataset, otherwise generate mock image data.    
if using_imagenet == True:  
    train_ds_input = tf.keras.utils.image_dataset_from_directory(
        '/direcotry/to/imagenet-mini/train',
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True,
        seed=1000,
        image_size=(image_size, image_size),
        subset=None)
else:
    train_ds_input = tf.data.Dataset.from_tensor_slices((np.random.rand(2000,image_size,image_size,3), np.random.randint(1000,size=2000))).cache().batch(batch_size)

#Initilize ResNet50 model
model = tf.keras.applications.resnet50.ResNet50(
    weights=None,
    input_shape=[image_size,image_size,3]
)

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.keras.optimizers.Adam(learning_rate = scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt,compression=hvd.Compression.fp16)
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=opt,experimental_run_tf_function=False)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]

# Train the model.
model.fit(train_ds_input, epochs=epoch_num, callbacks=callbacks, batch_size=batch_size)
