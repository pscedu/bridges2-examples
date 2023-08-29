import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd
from tensorflow.python.keras import optimizers
from timeit import default_timer as timer

from tqdm import tqdm
import numpy as np
import time

device = 'GPU'
# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
#gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.list_physical_devices('GPU')
print('number of GPUs:',len(gpus))
print(gpus,hvd.size())
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = argparse.ArgumentParser(description='Input values.')
parser.add_argument('--bz', type=int, default=128, help='batch size')

args = parser.parse_args()
batch_size = args.bz
print('batch size=',batch_size)

train_ds_input = tf.keras.utils.image_dataset_from_directory(
    '/ocean/projects/pscstaff/mwang7/data/imagenet/imagenet-mini/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=True,
    seed=1000,
    validation_split=None,
    subset=None)

#builder = tfds.ImageFolder('/ocean/projects/pscstaff/mwang7/data/imagenet/imagenet-mini/train')
#print(builder.info)  # num examples, labels... are automatically calculated
#ds = builder.as_dataset(split='train', shuffle_files=True)

# Pin GPU to be used to process local rank (one GPU per process)
#if torch.cuda.is_available():
#    torch.cuda.set_device(hvd.local_rank())

model = tf.keras.applications.resnet50.ResNet50(
    weights=None,
    input_shape=[256,256,3]
)

opt = tf.keras.optimizers.experimental.SGD(learning_rate = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
opt = hvd.DistributedOptimizer(opt)
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=opt)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
]

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
             (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.bz * len(train_ds_input) / time
        print('Iter #%d: %.1f img/sec per GPU' % (epoch, img_sec))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)

# Horovod: write logs on worker 0.
#if hvd.rank() == 0:
timing = TimingCallback()
callbacks.append(timing)


model.fit(train_ds_input, epochs=2,callbacks=callbacks,batch_size=batch_size)
