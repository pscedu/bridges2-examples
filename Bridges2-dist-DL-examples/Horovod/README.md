# An example of Running Horovod on Bridges-2

Here we show examples of running Horovod with [Tensorflow](#Tensorflow) and [PyTorch](#PyTorch).
This examples that train ResNet50 models with either Imagenet/Imagenet-mini dataset or dummy image data generated with random pixels.
A slurm script is included to show how to set up the environment and structure the slurm script for running on Bridges-2 GPU nodes.

## Data
If the flag `-imagenet` is set, it will read the Imagenet dataset from the specified directory. 

For Imagenet, you can access the dataset on Bridges-2 on `/ocean/datasets/community/imagenet`.

For Imagenet-mini, it can be download from this [website](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).

If the flag `-imagenet` is not set (default), it will generate mock images with random pixels. 

## Tensorflow
### Usage
```bash
Usage: tf_horovod.py  [-h] [-bz BATCH_SIZE] 
                           [-image_size IMAGE_SIZE]
                           [-epoch_num EPOCH_NUM]
                           [-mp]
                           [-imagenet]


Optional arguments:
  -bz BATCH_SIZE           Sepcify the data batch size per replica (default: 128).
  -image_size IMAGE_SIZE   Resize the image size to be IMAGE_SIZE x IMAGE_SIZE  (default: 128).
  -epoch_num EPOCH_NUM     Number of training epochs (default: 5).
  -mp                      Enable mixed precision training or not (default: False).
  -imagenet                Using Imagenet dataset for train or dummy data generated with random pixels (default: False). 
                
```
### Single Node
#### Interactive Session
Here we give example commands of running this example with 4 GPUs (node type not specified) with the GPU-shared partition for an hour:

##### With `singularity exec`
```bash
interact --partition GPU-shared --gres=gpu:v100:4
singularity exec --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif horovodrun -np 4 python3 tf_horovod.py
```

##### With `singularity shell`
```bash
interact --partition GPU-shared --gres=gpu:v100:4
singularity shell --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif
horovodrun -np 4 python3 tf_horovod.py
```
#### Batch Mode
After modifying the script to include the correct allocation account and working directory, submit the batch job by typing:
```bash
sbatch tf_horovod_single_node.sbatch
```

### Multi-nodes
#### Interactive Session
Here we give example commands of running this example with 2 GPU nodes (16 GPUs) with the GPU partition for an hour:

##### With `singularity exec`
```bash
interact --partition GPU --nodes 2 --gres=gpu:v100:8
mpirun -np 16 -H node_1:8,node_2:8 -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib singularity exec --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif python3 tf_horovod.py
```
Please replace `node_1` and `node_2` with the node names (e.g., v005) assigned for your sessions.

##### With `singularity shell`
```bash
interact --partition GPU --nodes 2 --gres=gpu:v100:8
singularity shell --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif
mpirun -np 16 -H node_1:8,node_2:8 -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3 tf_horovod.py
```
Please replace `node_1` and `node_2` with the node names (e.g., v005) assigned for your sessions.
#### Batch Mode
After modifying the script to include the correct allocation account and working directory, submit the batch job by typing:
```bash
sbatch tf_horovod_multi_nodes.sbatch
```

## PyTorch
### Usage
```bash
Usage: pt_horovod.py  [-h] [-bz BATCH_SIZE] 
                           [-image_size IMAGE_SIZE]
                           [-epoch_num EPOCH_NUM]
                           [-mp]
                           [-imagenet]


Optional arguments:
  -bz BATCH_SIZE           Sepcify the data batch size per replica (default: 128).
  -image_size IMAGE_SIZE   Resize the image size to be IMAGE_SIZE x IMAGE_SIZE  (default: 128).
  -epoch_num EPOCH_NUM     Number of training epochs (default: 5).
  -mp                      Enable mixed precision training or not (default: False).
  -imagenet                Using Imagenet dataset for train or dummy data generated with random pixels (default: False). 
                
```
### Single Node
#### Interactive Session
Here we give example commands of running this example with 4 GPUs (node type not specified) with the GPU-shared partition for an hour:

##### With `singularity exec`
```bash
interact --partition GPU-shared --gres=gpu:v100:4
singularity exec --nv /ocean/containers/horovod/horovod_latest.sif horovodrun -np 4 python3 pt_horovod.py
```

##### With `singularity shell`
```bash
interact --partition GPU-shared --gres=gpu:v100:4
singularity shell --nv /ocean/containers/horovod/horovod_latest.sif
horovodrun -np 4 python3 pt_horovod.py
```
#### Batch Mode
After modifying the script to include the correct allocation account and working directory, submit the batch job by typing:
```bash
sbatch pt_horovod_single_node.sbatch
```

### Multi-nodes
#### Interactive Session
Here we give example commands of running this example with 2 GPU nodes (16 GPUs) with the GPU partition for an hour:

##### With `singularity exec`
```bash
interact --partition GPU --nodes 2 --gres=gpu:v100:8
mpirun -np 16 -H node_1:8,node_2:8 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib singularity exec --nv /ocean/containers/horovod/horovod_latest.sif python3 pt_horovod.py
```
Please replace `node_1` and `node_2` with the node names (e.g., v005) assigned for your sessions.

##### With `singularity shell`
```bash
interact --partition GPU --nodes 2 --gres=gpu:v100:8
singularity shell --nv /ocean/containers/horovod/horovod_latest.sif
mpirun -np 16 -H node_1:8,node_2:8 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3 pt_horovod.py
```
Please replace `node_1` and `node_2` with the node names (e.g., v005) assigned for your sessions.
#### Batch Mode
After modifying the script to include the correct allocation account and working directory, submit the batch job by typing:
```bash
sbatch pt_horovod_multi_nodes.sbatch
```



