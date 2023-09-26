#!/usr/bin/bash
#SBATCH --job-name=horovod_TF_resnet50_2N_16GPU                 # Job name
#SBATCH --partition=GPU                                         # Partition
#SBATCH --time=0:30:00                                          # Walltime (HH:MM:SS)
#SBATCH --nodes=2                                               # Number of nodes
#SBATCH --gres=gpu:8                                    # Number of GPUs per node, including GPU type
#SBATCH --account=pscstaff                                      # Allocation to use resources from

cp ${0} slurm-${SLURM_JOB_ID}.sbatch

set -x

cd $HOME/_projects/b2/multinode/bridges2-examples/resnet50_imagenet_horovod/2-nodes-16-gpus

module load openmpi

####
# Use sacct to get the gres resources allocated
TRES_OUTPUT=$(sacct -j $SLURM_JOB_ID --format=AllocTRES --allocations --noheader -P)
# Use sacct to get the list of nodes allocated
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ',' | sed 's/,$//')

# Get the comma-separated list of nodes assigned to the job with GPU counts
export NODELIST_WITH_GPUS="$(python3 - ${NODELIST} ${TRES_OUTPUT} <<'EOF'
import sys
node_list = sys.argv[1].split(',')
tres_chunks = sys.argv[2].split(',')
for tres_chunk in tres_chunks:
    if tres_chunk.startswith('gres/gpu='):
        gres_gpu = tres_chunk.split('=')
        total_gpus = int(gres_gpu[-1])
    if tres_chunk.startswith('node='):
        gres_nodes = tres_chunk.split('=')
        total_nodes = int(gres_nodes[-1])
gpus_per_node = int(total_gpus / total_nodes)
if len(node_list) > 1:
    list_aggregation = [f'{node}:{gpus_per_node}' for node in node_list]
    print(",".join(list_aggregation))
else:
    for node in node_list:
        print(f'{node}:{gpus_per_node}')

EOF
)"

# Print the node list with GPU counts
echo "Nodes assigned to the job with GPU counts: $NODELIST_WITH_GPUS"
####

singularity exec --nv --bind $PROJECT,/ocean/datasets /ocean/containers/ngc/tensorflow/tensorflow_latest.sif pip install tensorflow_datasets
mpirun -np 16 -H $NODELIST_WITH_GPUS singularity exec --nv /ocean/containers/ngc/tensorflow/latest.sif python3 resnet50_imagenet_horovod.py