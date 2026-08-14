#!/bin/bash
#SBATCH --job-name=simbench_base_verb
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

export HF_CACHE_LOCAL=$HF_HOME
export LOCAL_WORKDIR=$PROJECT_DIR/eval/simbench
export OMP_NUM_THREADS=4

cd $PROJECT_DIR/eval/simbench

python generate_answers.py \
    --input_file SimBenchTargets_US.pkl \
    --output_file results/base_llama_verb_us.pkl \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --method verbalized
