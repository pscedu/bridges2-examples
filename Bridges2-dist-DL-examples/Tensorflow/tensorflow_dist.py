import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time 

import argparse

parser = argparse.ArgumentParser(description='Input values.')
parser.add_argument('-bz', type=int, default=128, help='Batch size per replica')
parser.add_argument('-image_size', type=int, default=128, help='Image size (number of pixels per size)')
parser.add_argument('-epoch_num', type=int, default=5, help='Number of training epochs')
parser.add_argument('-mp', action='store_true', help='Mixed precision training')
parser.add_argument('-imagenet', action='store_true', help='Using Imagenet dataset')

args = parser.parse_args()
batch_size_per_replica = args.bz        # batch size per replica
image_size = args.image_size
mixed_precision = args.mp
using_imagenet = args.imagenet
epoch_num = args.epoch_num

print('Batch size=',batch_size_per_replica,' (per replica)')
print('Image size=',image_size)
print('Mixed precision training=',mixed_precision)
print("Number of GPUs=", len(tf.config.list_physical_devices('GPU')))

mirrored_strategy = tf.distribute.MirroredStrategy()
batch_size = mirrored_strategy.num_replicas_in_sync * batch_size_per_replica

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

with mirrored_strategy.scope():
    model = tf.keras.applications.resnet50.ResNet50(weights=None,input_shape=[image_size,image_size,3])
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(train_ds_input, epochs=epoch_num)

