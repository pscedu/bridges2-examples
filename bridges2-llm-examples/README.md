# Bridges-2 Examples for Open Large Language Models

Here, we give some simple examples to start with Llama-2/Llama-3 and Gemma. Those examples are modified versions of examples in the [Llama-recipes](https://github.com/meta-llama/llama-recipes) and [Gemma website](https://ai.google.dev/gemma/docs) to accommodate machine configurations on Bridges-2. The configurations assume running jobs with NVIDIA V100 32GB GPUs (Bridges-2 V100-32 nodes). We also give some useful tips for running on Bridges-2, such as setting up a cache directory or addressing common issues with the environment setup.

## Llama

In the [Llama](Llama/) folder, we show examples of doing finetuning/inference with Llama-2 7B/Llama-3 8B using LoRA, with either a single GPU or two GPUs. The example is in Jupyter notebook format, which can be run on Bridges-2 using [OnDemand](https://ondemand.bridges2.psc.edu/). Depending on the configurations, finetuning Llama-2 7B can be done with a single V100 32GB GPU using LoRA and quantization with INT8, and finetuning Llama-3 8B will often require two V100 32GB GPUs with LoRA. We also show examples of how to run the [Llama-recipes](https://github.com/meta-llama/llama-recipes) using command lines via batch/interactive mode.

## Gemma
In the [Gemma](Gemma/) folder, we show examples of doing finetuning/inference with Llama-2 7B/Llama-3 8B using LoRA, with either a single GPU or two GPUs. The example is in Jupyter Notebook format, which can be run on Bridges-2 using [OnDemand](https://ondemand.bridges2.psc.edu/). Depending on the configurations, finetuning Gemma 2B can be done with a single V100 32GB GPU using LoRA and quantization with INT8, and finetuning Gemma 7B will often require two V100 32GB GPUs.

## Helpful tips for running on Bridges-2
- It is recommended that you download the model weights to your `$PROJECT` directory rather than the `$HOME` directory. The `$HOME` directory, by default, has a 25GB quota, which will quickly run out of storage when downloading large files such as LLM model weights. You can check your storage quota usage by typing `my_quotas` command.
- It is recommended to create a symlink for `.cache` folder, which by default located in your `$HOME` directory, to avoid exceeding storage quota limitation. For example, if you use the `kagglehub` API to download models, by default, the downloaded files will be placed in the `.cache` folder, the same as `huggingface.` To do so, for example, please type:
   ```
   mv $HOME/.cache $PROJECT/
   ln -s $PROJECT/.cache $HOME/.cache
   ```
- When you pip install llama-recipes inside an NGC container shell, it may install a local copy of the PyTorch library. Many times, it may interfere with your PyTorch library in the container. You can type `pip uninstall torch` to uninstall the local copy to avoid some issues.
