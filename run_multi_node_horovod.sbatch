#!/usr/bin/bash
#SBATCH -p GPU
#SBATCH -t 30:00
#SBATCH -N 2
#SBATCH --res=PSC 
#SBATCH --gpus=v100-32:16
#SBATCH -J tf-gpus
#SBATCH --account=pscstaff
#SBATCH --output=tf_horovod_two_node_imagenent_bz_128_r3.out

cp ${0} slurm-${SLURM_JOB_ID}.sbatch

set -x

cd /ocean/projects/pscstaff/mwang7/multi_gpu_test/Tensorflow

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
singularity exec --nv --bind /ocean/projects/pscstaff/mwang7,/ocean/datasets /ocean/containers/ngc/tensorflow/tensorflow_latest.sif pip install tensorflow_datasets
singularity exec --nv --bind /ocean/projects/pscstaff/mwang7,/ocean/datasets /ocean/containers/ngc/tensorflow/tensorflow_latest.sif horovodrun -np 8 -H v005:8 python resnet50_imagenet_horovod.py

