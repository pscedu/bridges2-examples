#!/usr/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:v100-32:2
#SBATCH -J llama-3-8b  #name of the job
#SBATCH --account=xxxxxxxx   #Please change it to your allocation ID

set -x

cd /ocean/projects/groupname/username/path-to-directory
singularity exec --nv /ocean/containers/ngc/pytorch/pytorch_latest.sif pip install llama-recipes
singularity exec --nv /ocean/containers/ngc/pytorch/pytorch_latest.sif \
      torchrun --nnodes 1 --nproc_per_node 2 -m llama_recipes.finetuning --enable_fsdp \
      --use_peft --peft_method lora --model_name /your_project_dir/llama-3-8b-hf/ \
      --batch_size_training 1 --output_dir /your_project_dir/llama-3-8b-hf-finetuned 
