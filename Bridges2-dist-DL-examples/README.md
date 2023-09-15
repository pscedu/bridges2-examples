# Bridges-2 examples for distributed deep learning training

Here we give simple examples that train ResNet50 models with either Imagenet/Imagenet-mini dataset or mock images generated with random pixels using a few different deep learning distributed training framework.

## PyTorch

Here we show an example of running PyTorch `DistributedDataParallel` framework in the [PyTorch](Pytorch/) section. 
A slurm script is included to show how to set up the environment and configuration to request resources for running on Bridges-2 GPU nodes. We also show example commands for running interactive sessions.

## Tensorflow
Here we show an example of running Tensorflow `tf.distributed.MirroredStrategy` framework in the [Tensorflow](Tensorflow/) section. 
A slurm script is included to show how to set up the environment and configuration to request resources for running on Bridges-2 GPU nodes. We also show example commands for running interactive sessions.

## Horovod
Here we show an example of running Horovod with Tensorflow in the [Horovod](Horovod/) section. 
A slurm script is included to show how to set up the environment and configuration to request resources for running on Bridges-2 GPU nodes. We also show example commands for running interactive sessions.
