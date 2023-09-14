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
parser.add_argument('-bz', type=int, default=64, help='Batch size')
parser.add_argument('-image_size', type=int, default=128, help='Image size')
parser.add_argument('-epoch_num', type=int, default=5, help='Number of training epochs')
parser.add_argument('-mp', action='store_true', help='Mixed Precision Training')
parser.add_argument('-imagenet', action='store_true', help='Mixed Precision Training')

args = parser.parse_args()
batch_size = args.bz
image_size = args.image_size
mixed_precision = args.mp
using_imagenet = args.imagenet
epoch_num = args.epoch_num
if hvd.local_rank() == 0:
    print('Batch size=',batch_size)
    print('Image size=',image_size)
    print('Number of Training Epochs:',epoch_num)
    print('Mixed precision training? ',mixed_precision)

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
    train_ds_input = tf.data.Dataset.from_tensor_slices((np.random.rand(2000,image_size,image_size,3), np.random.randint(1000,size=2000,dtype=tf.int64))).cache().repeat().batch(batch_size)

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

# Callback to calculate training throughput.
class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
             (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

    def on_batch_begin(self, atchb, logs=None):
        self.starttime = timer()

    def on_batch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.bz  / time
        self.img_secs.append(img_sec)

timing = TimingCallback()
callbacks.append(timing)

# Train the model.
model.fit(train_ds_input, epochs=epoch_num, callbacks=callbacks, batch_size=batch_size)
