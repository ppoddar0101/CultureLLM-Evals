#!/bin/bash
#SBATCH --job-name=prism_pairwise
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5   # 6 models, 1 seed each (flip augmentation handles position bias)

###############################################################################
# CONFIGURATION — UPDATE ADAPTER PATHS
###############################################################################
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

ADAPTER_US="$PROJECT_DIR/results/llama8b-english-wvq-morality"
ADAPTER_FRANCE="$PROJECT_DIR/models/culturellm-france-8b-morality"
ADAPTER_BRAZIL="$PROJECT_DIR/models/culturellm-brazil-8b-morality"
ADAPTER_INDIA="$PROJECT_DIR/models/culturellm-india-8b-morality"
ADAPTER_ITALY="$PROJECT_DIR/models/culturellm-italy-8b-morality"
###############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm
pip install datasets scipy --quiet

export HF_CACHE_LOCAL=$HF_HOME
export OMP_NUM_THREADS=4

cd $PROJECT_DIR

mkdir -p logs
mkdir -p prism_results

TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "============================================================"
echo "SLURM Array Task: ${TASK_ID}"
echo "============================================================"

case ${TASK_ID} in
    0)
        echo "Running CultureLLM-US"
        python prism_pairwise.py \
            --model_name "${BASE_MODEL}" \
            --adapter_path "${ADAPTER_US}" \
             --batch_size 8 \
            --output_dir prism_results
        ;;
    1)
        echo "Running CultureLLM-France"
        python prism_pairwise.py \
            --model_name "${BASE_MODEL}" \
            --adapter_path "${ADAPTER_FRANCE}" \
            --batch_size 8 \
            --output_dir prism_results
        ;;
    2)
        echo "Running CultureLLM-Brazil"
        python prism_pairwise.py \
            --model_name "${BASE_MODEL}" \
            --adapter_path "${ADAPTER_BRAZIL}" \
            --batch_size 8 \
            --output_dir prism_results
        ;;
    3)
        echo "Running CultureLLM-Italy"
        python prism_pairwise.py \
            --model_name "${BASE_MODEL}" \
            --adapter_path "${ADAPTER_ITALY}" \
            --batch_size 8 \
            --output_dir prism_results
        ;;
    4)
        echo "Running CultureLLM-India"
        python prism_pairwise.py \
            --model_name "${BASE_MODEL}" \
            --adapter_path "${ADAPTER_INDIA}" \
            --batch_size 8 \
            --output_dir prism_results
        ;;
    5)
        echo "Running Base Llama-3.1-8B-Instruct"
        python prism_pairwise.py \
            --model_name "${BASE_MODEL}" \
            --batch_size 8 \
            --output_dir prism_results
        ;;
esac

echo "Done: task ${TASK_ID}"