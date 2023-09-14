#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 2 
#SBATCH --gpus=v100-32:16              # request two nodes, each with 8 GPUs. So 16 GPUs in total. 
#SBATCH -J tf_horovod_multi_nodes      # Name of the job
#SBATCH --account=xxxxxxx              # Please change it to your allocation ID

set -x
cd /ocean/projects/groupname/username/path-to-directory

#running with a NGC container
singularity exec --nv --bind /ocean/projects/pscstaff/mwang7,/ocean/datasets /ocean/containers/ngc/tensorflow/tensorflow_latest.sif pip install tensorflow_datasets
mpirun -np 16 -H v005:8,v004:8 singularity exec --nv /ocean/containers/ngc/tensorflow/latest.sif python3 resnet50_imagenet_horovod.py
