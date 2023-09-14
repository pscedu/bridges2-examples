#!/usr/bin/bash
#SBATCH -p GPU
#SBATCH -t 30:00
#SBATCH -N 2
#SBATCH --res=PSC 
#SBATCH --gpus=v100-32:16
#SBATCH --nodelist v007,v005
#SBATCH -J tf-gpus
#SBATCH --account=pscstaff
#SBATCH --output=tf_horovod_two_node_imagenent_bz_128_r4.out

cp ${0} slurm-${SLURM_JOB_ID}.sbatch

set -x

cd /ocean/projects/pscstaff/mwang7/multi_gpu_test/Tensorflow

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
singularity exec --nv --bind /ocean/projects/pscstaff/mwang7,/ocean/datasets /ocean/containers/ngc/tensorflow/tensorflow_latest.sif pip install tensorflow_datasets
singularity exec --nv --bind /ocean/projects/pscstaff/mwang7,/ocean/datasets /ocean/containers/ngc/tensorflow/tensorflow_latest.sif horovodrun -np 16 -H v005:8,v007:8 python resnet50_imagenet_horovod.py

# 2023-09-14 Julian
#    julian@br013:~ (08:42 AM) $ interact --nodes 2 --partition GPU --gres=gpu:8
#
#        A command prompt will appear when your session begins
#        "Ctrl+d" or "exit" will end your session
#
#        --gres=gpu:8 --nodes=2 --partition=GPU
#        salloc -J Interact --gres=gpu:8 --nodes=2 --partition=GPU
#        salloc: Pending job allocation 19058433
#        salloc: job 19058433 queued and waiting for resources
#        salloc: job 19058433 has been allocated resources
#        salloc: Granted job allocation 19058433
#        salloc: Waiting for resource configuration
#        salloc: Nodes v[030-031] are ready for job
#
#        julian@v030:~ (08:43 AM) $
#
#    module load openmpi
#
#    mpirun -np 2 -H v030:1,v031:1 hostname
#        v030.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#
#    mpirun -np 16 -H v030:8,v031:8 hostname #  singularity exec --nv /ocean/containers/ngc/tensorflow/latest.sif hostname
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v030.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#
#    mpirun -np 16 -H v030:8,v031:8 singularity exec /ocean/containers/ngc/tensorflow/latest.sif hostname
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        v030.ib.bridges2.psc.edu
#        v031.ib.bridges2.psc.edu
#
#    mpirun -np 16 -H v030:8,v031:8 singularity exec --nv /ocean/containers/ngc/tensorflow/latest.sif python3 resnet50_imagenet_horovod.py
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        13:4: not a valid test operator: (
#        13:4: not a valid test operator: 525.60.13
#        2023-09-14 08:59:36.314267: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.315404: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.327153: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.343578: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.346749: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.374336: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.377961: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.386903: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.595711: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.600281: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.600330: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.610430: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.614039: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.615577: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.615989: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        2023-09-14 08:59:36.620072: I tensorflow/core/platform/cpu_feature_guard.cc:183] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#        To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        number of GPUs: 8
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        batch size= 128
#        batch size= 128
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        number of GPUs: 8
#        batch size= 128
#        batch size= 128
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        number of GPUs: 8
#        number of GPUs: 8
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        batch size= 128
#        batch size= 128
#        [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:7', device_type='GPU')] 16
#        batch size= 128
#        batch size= 128
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        2023-09-14 08:59:54.300966: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 4, name: Tesla V100-SXM2-16GB, pci bus id: 0000:89:00.0, compute capability: 7.0
#        2023-09-14 08:59:54.345019: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 3, name: Tesla V100-SXM2-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0
#        2023-09-14 08:59:54.348343: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 1, name: Tesla V100-SXM2-16GB, pci bus id: 0000:16:00.0, compute capability: 7.0
#        2023-09-14 08:59:54.353562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 5, name: Tesla V100-SXM2-16GB, pci bus id: 0000:8a:00.0, compute capability: 7.0
#        2023-09-14 08:59:54.368439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 6, name: Tesla V100-SXM2-16GB, pci bus id: 0000:b2:00.0, compute capability: 7.0
#        2023-09-14 08:59:54.370131: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:15:00.0, compute capability: 7.0
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        Found 34745 files belonging to 1000 classes.
#        2023-09-14 08:59:54.381354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 7, name: Tesla V100-SXM2-16GB, pci bus id: 0000:b3:00.0, compute capability: 7.0
#        2023-09-14 08:59:54.390154: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 2, name: Tesla V100-SXM2-16GB, pci bus id: 0000:3a:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.431331: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 2, name: Tesla V100-SXM2-16GB, pci bus id: 0000:3a:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.518813: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 5, name: Tesla V100-SXM2-16GB, pci bus id: 0000:8a:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.522035: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 1, name: Tesla V100-SXM2-16GB, pci bus id: 0000:16:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.541746: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 3, name: Tesla V100-SXM2-16GB, pci bus id: 0000:3b:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.542762: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 4, name: Tesla V100-SXM2-16GB, pci bus id: 0000:89:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.559853: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 6, name: Tesla V100-SXM2-16GB, pci bus id: 0000:b2:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.571477: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 7, name: Tesla V100-SXM2-16GB, pci bus id: 0000:b3:00.0, compute capability: 7.0
#        2023-09-14 08:59:55.576131: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1638] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14777 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:15:00.0, compute capability: 7.0
#        2023-09-14 08:59:56.667129: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        2023-09-14 08:59:56.667425: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.672236: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        2023-09-14 08:59:56.672521: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.700892: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.701187: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.702876: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.703164: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.704153: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.704454: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.721164: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.721446: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        Epoch 1/2
#        Epoch 1/2
#        2023-09-14 08:59:56.754442: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.754730: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        Epoch 1/2
#        Epoch 1/2
#        Epoch 1/2
#        Epoch 1/2
#        Epoch 1/2
#        2023-09-14 08:59:56.859226: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:56.859556: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        Epoch 1/2
#        2023-09-14 08:59:57.806012: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.806310: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        2023-09-14 08:59:57.808294: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.808577: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.823418: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.823702: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.833005: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        2023-09-14 08:59:57.833293: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        2023-09-14 08:59:57.834967: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.835260: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [34745]
#             [[{{node Placeholder/_0}}]]
#        2023-09-14 08:59:57.838547: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.838833: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        Epoch 1/2
#        Epoch 1/2
#        Epoch 1/2
#        2023-09-14 08:59:57.899023: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.899312: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        Epoch 1/2
#        Epoch 1/2
#        Epoch 1/2
#        Epoch 1/2
#        2023-09-14 08:59:57.966537: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        2023-09-14 08:59:57.966864: I tensorflow/core/common_runtime/executor.cc:1209] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [34745]
#             [[{{node Placeholder/_4}}]]
#        Epoch 1/2
#        2023-09-14 09:00:35.753058: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:35.869839: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:35.889014: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:35.890575: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:36.012404: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:36.162684: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:36.308673: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 306 of 1024
#        2023-09-14 09:00:36.414691: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 310 of 1024
#        2023-09-14 09:00:36.682504: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:37.212436: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 316 of 1024
#        2023-09-14 09:00:37.225327: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 316 of 1024
#        2023-09-14 09:00:37.292629: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 318 of 1024
#        2023-09-14 09:00:37.534266: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 325 of 1024
#        2023-09-14 09:00:37.586752: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:00:37.881857: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 338 of 1024
#        2023-09-14 09:00:38.500256: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 352 of 1024
#        2023-09-14 09:00:44.453331: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.574577: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.574652: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.725978: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.726049: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.792812: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.813747: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.868247: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.912976: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.913041: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.928980: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.929028: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.938240: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.984979: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.985026: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:44.990160: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.036976: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.037037: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.052918: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.052983: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.061978: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.062024: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.110976: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.111020: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.119984: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.120037: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.134739: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.171786: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.188974: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.189033: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.235007: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.235056: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.246283: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.249977: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.250020: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.275984: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.295211: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.295260: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.317624: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.317659: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.360043: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.360088: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.366181: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.366222: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.397555: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.397598: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.427693: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.427741: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.434935: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.434982: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.444676: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.444725: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.447769: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.502265: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.502327: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.522975: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.523018: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.568981: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.569030: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.584564: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.584612: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.584976: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.585020: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.585177: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.654994: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.693189: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.693241: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.695949: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.695996: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.763910: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.763963: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.795419: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.835645: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.835698: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.891646: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.891692: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.910886: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:45.910952: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:46.054986: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:46.055045: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:00:46.238937: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 503 of 1024
#        2023-09-14 09:00:46.408616: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 508 of 1024
#        2023-09-14 09:00:46.812659: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 515 of 1024
#        2023-09-14 09:00:47.121312: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 530 of 1024
#        2023-09-14 09:00:47.299923: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 539 of 1024
#        2023-09-14 09:00:47.516023: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 545 of 1024
#        2023-09-14 09:00:47.899314: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 554 of 1024
#        2023-09-14 09:00:48.490751: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 569 of 1024
#        2023-09-14 09:00:56.305108: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 841 of 1024
#        2023-09-14 09:00:56.413896: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 846 of 1024
#        2023-09-14 09:00:56.845769: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 848 of 1024
#        2023-09-14 09:00:57.136764: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 852 of 1024
#        2023-09-14 09:00:57.312795: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 857 of 1024
#        2023-09-14 09:00:57.683979: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 866 of 1024
#        2023-09-14 09:00:57.895523: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 873 of 1024
#        2023-09-14 09:00:58.490017: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:392] Filling up shuffle buffer (this may take a while): 895 of 1024
#        2023-09-14 09:01:02.780245: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780258: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780259: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780326: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780327: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780333: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780355: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:02.780468: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:417] Shuffle buffer filled.
#        2023-09-14 09:01:09.088832: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.088872: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.108768: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.112413: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.122322: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.180528: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.180562: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:09.198270: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8902
#        2023-09-14 09:01:18.683513: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.804850: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.804921: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.855524: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.922864: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.922914: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.976868: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:18.976918: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.103849: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.103893: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.117875: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.125847: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.225530: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.239855: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.239903: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.240540: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.240587: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.268125: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.316851: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.345166: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.345213: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.360541: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.360586: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.378846: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.378906: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.395136: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.395206: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.430751: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.430797: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.432172: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.446872: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.473415: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.473460: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.533875: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.533930: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.560853: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.560905: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.569533: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.569583: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.571852: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.571912: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.585502: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.660640: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.684958: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.689851: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.689904: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.691521: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.691570: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.700854: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.700904: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.760856: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.778854: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.778920: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.807853: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.807900: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.836040: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.836086: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.872793: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.872851: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.892018: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.904854: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.904903: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.908941: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.937860: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:19.937911: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.005855: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.005912: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.008869: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.008923: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.036130: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.036179: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.113994: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.59GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.156862: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.156917: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.166036: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.166086: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.234144: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.234202: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.356858: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:20.356921: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.82GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
#        2023-09-14 09:01:30.521395: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x15390e732f40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.521432: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.522462: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x152724747eb0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.522496: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.528455: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x1450ca278940 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.528485: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.529967: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14ef1dedc600 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.529999: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.536267: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14c4f5e99b30 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.536298: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.538378: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14ac45be4490 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.538412: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.539320: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14d39defeb10 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.539352: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.542686: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x1458c7910360 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.542718: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.550194: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x146c9322a590 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.550224: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.556341: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14af78c8e1e0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.556369: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.557403: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x149f4f31ef40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.557435: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.559297: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x145f7fe4e3a0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.559330: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.580204: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x148e5bf8cc20 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.580230: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.627148: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14d80d8ff1a0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.627174: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:30.974176: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x14725779b740 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:30.974211: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:31.197217: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x1475736f42b0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
#        2023-09-14 09:01:31.197249: I tensorflow/compiler/xla/service/service.cc:177]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
#        2023-09-14 09:01:31.741450: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.742287: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.745569: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.745735: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.746522: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.749087: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.749528: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.751503: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.751779: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.752408: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.752667: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.752826: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.753959: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.754557: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:31.819698: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#        2023-09-14 09:01:32.069796: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
#         72/272 [======>.......................] - ETA: 11:26 - loss: 7.2847
#
#    GPU usage. v030 on the left, v031 on the right
#        julian@v030:~ (08:59 AM) $ gpustat                                           a   julian@v031:~ (09:01 AM) $ gpustat
#        v030.ib.bridges2.psc.edu  Thu Sep 14 08:59:58 2023  525.60.13                a   v031.ib.bridges2.psc.edu  Thu Sep 14 09:01:11 2023  525.60.13
#        [0] Tesla V100-SXM2-16GB | 31°C,   0 % |   439 / 16384 MB | julian(436M)     a   [0] Tesla V100-SXM2-16GB | 33°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [1] Tesla V100-SXM2-16GB | 34°C,   0 % |   439 / 16384 MB | julian(436M)     a   [1] Tesla V100-SXM2-16GB | 33°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [2] Tesla V100-SXM2-16GB | 30°C,   0 % |   567 / 16384 MB | julian(564M)     a   [2] Tesla V100-SXM2-16GB | 30°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [3] Tesla V100-SXM2-16GB | 33°C,   0 % |   567 / 16384 MB | julian(564M)     a   [3] Tesla V100-SXM2-16GB | 35°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [4] Tesla V100-SXM2-16GB | 31°C,   0 % |   567 / 16384 MB | julian(564M)     a   [4] Tesla V100-SXM2-16GB | 32°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [5] Tesla V100-SXM2-16GB | 32°C,   0 % |   567 / 16384 MB | julian(564M)     a   [5] Tesla V100-SXM2-16GB | 35°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [6] Tesla V100-SXM2-16GB | 30°C,   0 % |   567 / 16384 MB | julian(564M)     a   [6] Tesla V100-SXM2-16GB | 31°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        [7] Tesla V100-SXM2-16GB | 31°C,   0 % |   567 / 16384 MB | julian(564M)     a   [7] Tesla V100-SXM2-16GB | 33°C,   0 % |  1347 / 16384 MB | julian(1344M)
#        julian@v030:~ (09:00 AM) $ gpustat                                           a   julian@v031:~ (09:01 AM) $ gpustat
#        v030.ib.bridges2.psc.edu  Thu Sep 14 09:01:03 2023  525.60.13                a   v031.ib.bridges2.psc.edu  Thu Sep 14 09:01:15 2023  525.60.13
#        [0] Tesla V100-SXM2-16GB | 36°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [0] Tesla V100-SXM2-16GB | 36°C,  44 % |  9549 / 16384 MB | julian(9546M)
#        [1] Tesla V100-SXM2-16GB | 38°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [1] Tesla V100-SXM2-16GB | 36°C,  51 % |  9549 / 16384 MB | julian(9546M)
#        [2] Tesla V100-SXM2-16GB | 35°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [2] Tesla V100-SXM2-16GB | 35°C,  49 % |  9549 / 16384 MB | julian(9546M)
#        [3] Tesla V100-SXM2-16GB | 37°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [3] Tesla V100-SXM2-16GB | 38°C,  45 % |  9549 / 16384 MB | julian(9546M)
#        [4] Tesla V100-SXM2-16GB | 34°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [4] Tesla V100-SXM2-16GB | 36°C,  53 % |  9549 / 16384 MB | julian(9546M)
#        [5] Tesla V100-SXM2-16GB | 37°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [5] Tesla V100-SXM2-16GB | 37°C,  62 % |  9549 / 16384 MB | julian(9546M)
#        [6] Tesla V100-SXM2-16GB | 34°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [6] Tesla V100-SXM2-16GB | 36°C,  49 % |  9549 / 16384 MB | julian(9546M)
#        [7] Tesla V100-SXM2-16GB | 35°C,   0 % | 15113 / 16384 MB | julian(15110M)   a   [7] Tesla V100-SXM2-16GB | 36°C,  52 % |  9549 / 16384 MB | julian(9546M)
#        julian@v030:~ (09:01 AM) $ gpustat                                           a   julian@v031:~ (09:01 AM) $ gpustat
#        v030.ib.bridges2.psc.edu  Thu Sep 14 09:01:40 2023  525.60.13                a   v031.ib.bridges2.psc.edu  Thu Sep 14 09:01:43 2023  525.60.13
#        [0] Tesla V100-SXM2-16GB | 34°C,   0 % | 15433 / 16384 MB | julian(15430M)   a   [0] Tesla V100-SXM2-16GB | 35°C,   0 % | 15433 / 16384 MB | julian(15430M)
#        [1] Tesla V100-SXM2-16GB | 37°C,   0 % | 15381 / 16384 MB | julian(15378M)   a   [1] Tesla V100-SXM2-16GB | 36°C,   0 % | 15381 / 16384 MB | julian(15378M)
#        [2] Tesla V100-SXM2-16GB | 33°C,   0 % | 15397 / 16384 MB | julian(15394M)   a   [2] Tesla V100-SXM2-16GB | 32°C,   0 % | 15397 / 16384 MB | julian(15394M)
#        [3] Tesla V100-SXM2-16GB | 36°C,   0 % | 15349 / 16384 MB | julian(15346M)   a   [3] Tesla V100-SXM2-16GB | 38°C,   0 % | 15349 / 16384 MB | julian(15346M)
#        [4] Tesla V100-SXM2-16GB | 33°C,   0 % | 15473 / 16384 MB | julian(15470M)   a   [4] Tesla V100-SXM2-16GB | 34°C,   0 % | 15473 / 16384 MB | julian(15470M)
#        [5] Tesla V100-SXM2-16GB | 36°C,   0 % | 15425 / 16384 MB | julian(15422M)   a   [5] Tesla V100-SXM2-16GB | 38°C,   0 % | 15425 / 16384 MB | julian(15422M)
#        [6] Tesla V100-SXM2-16GB | 32°C,   0 % | 15309 / 16384 MB | julian(15306M)   a   [6] Tesla V100-SXM2-16GB | 33°C,   0 % | 15309 / 16384 MB | julian(15306M)
#        [7] Tesla V100-SXM2-16GB | 34°C,   0 % | 15337 / 16384 MB | julian(15334M)   a   [7] Tesla V100-SXM2-16GB | 36°C,   0 % | 15337 / 16384 MB | julian(15334M)
#
#        272/272 [==============================] - ETA: 0s - loss: 7.0031Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.6 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        Iter #0: 33.5 img/sec per GPU
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1037s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        272/272 [==============================] - 1038s 3s/step - loss: 7.0030
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#        Epoch 2/2
#         24/272 [=>............................] - ETA: 4:00 - loss: 6.7237
