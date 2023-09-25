#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=v100:4
#SBATCH -J pt-horovod-single-node          # name of the job
#SBATCH --account=xxxxxxx     #Please change it to your allocation ID

set -x
cd /ocean/projects/groupname/username/path-to-directory

#running with a NGC container
singularity exec --nv /ocean/containers/horovod/horovod_latest.sif horovodrun -np 4 python3 pt_horovod.py
