#!/bin/bash
#SBATCH --job-name=prb_pairwise
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-17   # 3 subsets x 2 models x 3 seeds = 18 jobs
 
###############################################################################
# CONFIGURATION — UPDATE THE ADAPTER PATH
###############################################################################
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
CULTURELLM_US_ADAPTER="$PROJECT_DIR/results/llama8b-english-wvq-morality"  # <-- SET THIS
###############################################################################
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

pip install datasets --quiet
 
export HF_CACHE_LOCAL=$HF_HOME
export OMP_NUM_THREADS=4
 
cd $PROJECT_DIR
 
mkdir -p logs
mkdir -p prb_results
 
# Define the grid
SUBSETS=(
    "Art_and_Entertainment"
    "Lifestyle_and_Personal_Development"
    "Society_and_Culture"
)
SEEDS=(42 123 456)
 
# Models: 0 = CultureLLM-US (with adapter), 1 = base (no adapter)
NUM_SUBSETS=3
NUM_MODELS=2
NUM_SEEDS=3
 
# Decode array task ID -> (subset_idx, model_idx, seed_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$((TASK_ID % NUM_SEEDS))
REMAINING=$((TASK_ID / NUM_SEEDS))
MODEL_IDX=$((REMAINING % NUM_MODELS))
SUBSET_IDX=$((REMAINING / NUM_MODELS))
 
SUBSET=${SUBSETS[$SUBSET_IDX]}
SEED=${SEEDS[$SEED_IDX]}
 
echo "============================================================"
echo "SLURM Array Task: ${TASK_ID}"
echo "  Subset: ${SUBSET}"
echo "  Model:  ${MODEL_IDX} (0=CultureLLM-US, 1=Base)"
echo "  Seed:   ${SEED}"
echo "============================================================"
 
if [ "${MODEL_IDX}" -eq 0 ]; then
    # CultureLLM-US with LoRA adapter
    python pairwise.py \
        --model_name "${BASE_MODEL}" \
        --adapter_path "${CULTURELLM_US_ADAPTER}" \
        --subset "${SUBSET}" \
        --seed "${SEED}" \
        --output_dir prb_results
else
    # Base Llama-3.1-8B-Instruct (no adapter)
    python pairwise.py \
        --model_name "${BASE_MODEL}" \
        --subset "${SUBSET}" \
        --seed "${SEED}" \
        --output_dir prb_results
fi
 
echo "Done: task ${TASK_ID}"