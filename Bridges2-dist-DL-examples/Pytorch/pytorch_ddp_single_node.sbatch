#!/usr/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH -J pytorch-resnet50  #name of the job
#SBATCH --account=xxxxxxxx   #Please change it to your allocation ID

set -x

cd /ocean/projects/groupname/username/path-to-directory

singularity exec --nv /ocean/containers/ngc/pytorch/pytorch_latest.sif torchrun --nnodes=1 --nproc_per_nod=4 pytorch_ddp.py

#Run with AI module enviroment
#module load AI/pytorch_23.02-1.13.1-py3 
#torchrun --nnodes=1 --nproc_per_nod=4 pytorch_ddp.py
