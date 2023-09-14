# Bridges-2 examples for distributed deep learning training

Here we give simple examples that train ResNet50 models with either Imagenet/Imagenet-mini dataset or mock images generated with random pixels using a few different deep learning distributed training framework.

## PyTorch

Here we show an example of running PyTorch `DistributedDataParallel` framework in [PyTorch](Pytorch/) section. 
A slurm script is included to show how to set up the environment and structure the slurm script for running on Bridges-2 GPU nodes.

## Tensorflow
Here we show an example of running Tensorflow `tf.distributed.MirroredStrategy` framework in [Tensorflow](Tensorflow/) section. 
A slurm script is included to show how to set up the environment and structure the slurm script  for running on Bridges-2 GPU nodes.

## Horovod
Here we show an example of running Horovod with Tensorflow in [Horovod](Horovod/) section. 
A slurm script is included to show how to set up the environment and structure the slurm script  for running on Bridges-2 GPU nodes.
