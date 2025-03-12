#!/bin/bash
#SBATCH --job-name=ai_amb
#SBATCH --output=/users/isj0001/AI_Ambassador/AIAmbassador/code/CRG/hpc/crg_hpc.out
#SBATCH --error=/users/isj0001/AI_Ambassador/AIAmbassador/code/CRG/hpc/crg_hpc.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=comm_gpu_inter

module purge
module load lang/gcc/12.2.0 lang/python/cpython_3.10.11_gcc122 parallel/cuda/11.8

source /users/isj0001/AI_Ambassador/AIAmbassador/code/CRG/venv/bin/activate
python3 /users/isj0001/AI_Ambassador/AIAmbassador/code/CRG/classify/traditional_ML.py --LR -force
