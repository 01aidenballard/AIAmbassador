#!/bin/bash
#SBATCH --job-name=finetune_transformers
#SBATCH --output=/scratch/isj0001/AIAmbassador/code/CRG/hpc/finetune_trans.out
#SBATCH --error=/scratch/isj0001/AIAmbassador/code/CRG/hpc/finetune_trans.err
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

source /scratch/isj0001/AIAmbassador/code/CRG/venv/bin/activate
python3 /scratch/isj0001/AIAmbassador/code/CRG/classify/finetune_transformer.py --force --BERT
