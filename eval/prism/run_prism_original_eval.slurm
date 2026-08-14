#!/bin/bash
#SBATCH --job-name=prism_gemini
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/prism_shard_%a_%j.out
#SBATCH --error=logs/prism_shard_%a_%j.err
#SBATCH --array=0-3

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

export GEMINI_API_KEYS="$GEMINI_API_KEYS"  # set this in your environment
export HF_CACHE_LOCAL=$HF_HOME
export OMP_NUM_THREADS=4

NUM_SHARDS=4

cd $PROJECT_DIR

#python eval_brazil.py \
#    --model_name meta-llama/Llama-3.1-8B-Instruct \
#    --adapter_path $PROJECT_DIR/models/culturellm-brazil-8b-morality \
#    --output_dir prism_original_results

python prism_gemini_resumable.py \
        --output_dir prism_original_results \
        --model gemini-2.5-flash \
        --concurrency 50 \
        --shard ${SLURM_ARRAY_TASK_ID} \
        --num_shards ${NUM_SHARDS}
