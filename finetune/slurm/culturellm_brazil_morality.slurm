#!/bin/bash
#SBATCH --job-name=culturellm_brazil_mor
#SBATCH --partition=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=c05-05,c06-05
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd $PROJECT_DIR

python llama_finetune_general.py train   --base_model "meta-llama/Llama-3.1-8B-Instruct"   --new_model "./models/culturellm-brazil-8b-morality"   --data_files "data/Brazil/Finetune/WVQ_Brazil_v2_llama_minus_morality.jsonl"   --country "Brazil"
