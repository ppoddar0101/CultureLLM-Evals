#!/bin/bash
#SBATCH --job-name=simbench_india_cult
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
    --input_file SimBenchTargets_India_pop.pkl \
    --output_file results/india_culturellm_verb.pkl \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter_path "$PROJECT_DIR/models/culturellm-india-8b-morality" \
    --method verbalized
    
#python generate_answers.py \
#    --input_file SimBenchTargets_India_pop.pkl \
#    --output_file results/india_base_verb.pkl \
#    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
#    --method verbalized
