#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=v100:4
#SBATCH -J tf-dist-resnet50          #name of the job
#SBATCH --account=xxxxxxx     #Please change it to your allocation ID

set -x
cd /ocean/projects/groupname/username/path-to-directory


#running with a NGC container
singularity exec --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif python tensorflow_dist.py

#running with AI module
#module load AI/tensorflow_23.02-2.10.0-py3 
#python tensorflow_dist.py
