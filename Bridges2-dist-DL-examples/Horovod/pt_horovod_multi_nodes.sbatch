#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 2 
#SBATCH --gpus=v100-32:16 
#SBATCH -t 30:00
#SBATCH -J pt_horovod_multi_nodes      # Name of the job
#SBATCH --account=xxxxxxx              # Please change it to your allocation ID

set -x
cd /ocean/projects/groupname/username/path-to-directory

module load openmpi

####
# Use sacct to get the list of nodes allocated
NODELIST=$(sacct -j $SLURM_JOB_ID --format=nodelist --allocations --noheader -P)

# Get the comma-separated list of nodes assigned to the job with GPU counts
export NODELIST_WITH_GPUS="$(python3 - ${NODELIST} <<'EOF'
import sys
node_list = sys.argv[1]

node_list = node_list.split(',')
if len(node_list) > 1:
    list_aggregation = [f'{node}:8' for node in node_list]
    print(",".join(list_aggregation))
else:
    for node in node_list:
        print(f'{node}:8')
EOF
)"

# Print the node list with GPU counts
echo "Nodes assigned to the job with GPU counts: $NODELIST_WITH_GPUS"
####

#running with a NGC container
mpirun -np 16 -H $NODELIST_WITH_GPUS -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib singularity exec --nv /ocean/containers/horovod/horovod_latest.sif python3 pt_horovod.py
