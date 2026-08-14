#!/bin/bash
#SBATCH --job-name=siqa_eval_gemini
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd $PROJECT_DIR

python -c "from google import genai; print('google-genai imported successfully')"

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="

echo ""
echo "====== GEMINI - VALIDATION ======"
python eval_social_iqa_gemini.py \
  --model gemini-2.5-flash \
  --split validation \
  --output results_siqa_val_gemini.json

echo ""
echo "====== GEMINI - TEST ======"
python eval_social_iqa_gemini.py \
  --model gemini-2.5-flash \
  --split test \
  --output results_siqa_test_gemini.json

echo ""
echo "====== ALL DONE ======"