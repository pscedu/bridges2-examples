# An example of Running Horovod Tensorflow on Bridges-2

Here we show an example of running Horovod with Tensorflow.
This examples that train ResNet50 models with either Imagenet/Imagenet-mini dataset or mock images generated with random pixels.
A slurm script is included to show how to set up the environment and structure the slurm script for running on Bridges-2 GPU nodes.

## Data
If the flag `-imagenet` is set to `True`, it will read the Imagenet dataset from the specified directory. 

For Imagenet, you can access the dataset on Bridges-2 on `/ocean/datasets/community/imagenet`.

For Imagenet-mini, it can be download from this [website](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).

If the flag `-imagenet` is set to `False` (default), it will generate mock images with random pixels. 

## Usage
```bash
Usage: tf_horovod.py  [-h] [-bz BATCH_SIZE] 
                           [-image_size IMAGE_SIZE]
                           [-epoch_num EPOCH_NUM]
                           [-mp MP]
                           [-imagenet IMAGENET]


Optional arguments:
  -bz BATCH_SIZE           Sepcify the data batch size (default: 128).
  -image_size IMAGE_SIZE   Resize the image size to be IMAGE_SIZE x IMAGE_SIZE  (default: 128).
  -epoch_num EPOCH_NUM     Number of training epochs (default: 5).
  -mp MP                   Enable mixed precision training or not (default: False).
  -imagenet IMAGENET       Using Imagenet dataset for train or mock data generated with random pixels (default: False). 
                
```

## Interactive Session
Here we give example commands of running this example with 4 GPUs (node type not specified) with the GPU-shared partition for an hour:

#### With `singularity exec`
```bash
interact --partition GPU-shared --gres=gpu:v100:4
singularity exec --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif horovodrun -np 4 tf_horovod.py
```

#### With `singularity shell`
```bash
interact --partition GPU-shared --gres=gpu:v100:4
singularity shell --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif
horovodrun -np 4 tf_horovod.py
```
## Batch Mode
After modifying the script to include the correct allocation account and working directory, submit the batch job by typing:
```bash
sbatch tf_horovod_single_node.sbatch
```


