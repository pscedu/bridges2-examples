import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time 

import argparse

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

print('Batch size=',batch_size)
print('Image size=',image_size)
print('Number of Training Epochs:',epoch_num)
print('Mixed precision training? ',mixed_precision)
print("number of GPUs: ", len(tf.config.list_physical_devices('GPU')))

mirrored_strategy = tf.distribute.MirroredStrategy()

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

class timer(tf.keras.callbacks.Callback):
    def __init__ (self): # initialization of the callback
        super(timer, self).__init__()
    def on_epoch_begin(self,epoch, logs=None):
        self.now= time.time()
    def on_epoch_end(self,epoch, logs=None): 
        later=time.time()
        duration=later-self.now 
        print('\nfor epoch ', epoch +1, ' Throughput:', batch_size*len(train_ds_input)/duration, ' [samples/s]')

with mirrored_strategy.scope():
    model = tf.keras.applications.resnet50.ResNet50(weights=None,input_shape=[image_size,image_size,3])
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(train_ds_input, epochs=epoch_num ,callbacks=[timer()])

