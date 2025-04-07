#!/bin/bash
#SBATCH --job-name=finetune_directllm
#SBATCH --output=/scratch/isj0001/AIAmbassador/code/DirectLLM/hpc/direct_llm.out
#SBATCH --error=/scratch/isj0001/AIAmbassador/code/DirectLLM/hpc/direct_llm.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=comm_gpu_inter
#SBATCH --mail-user=isj0001@mix.wvu.edu
#SBATCH --mail-type=end

module purge
module load lang/gcc/12.2.0 lang/python/cpython_3.10.11_gcc122 parallel/cuda/11.8

source /scratch/isj0001/AIAmbassador/code/DirectLLM/venv/bin/activate
python3 /scratch/isj0001/AIAmbassador/code/DirectLLM/flan_t5.py --train --hpc
