#!/bin/bash
#SBATCH --job-name=siqa_eval
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm
 
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
 
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="
 
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
 
cd $PROJECT_DIR
 
# ── Base model on validation ──
echo ""
echo "====== BASE MODEL - VALIDATION ======"
python eval_social_iqa.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --split validation \
  --output results_siqa_val_base.json
 
# ── US CultureLLM on validation ──
echo ""
echo "====== US CULTURELLM - VALIDATION ======"
python eval_social_iqa.py \
  --model $PROJECT_DIR/results/llama8b-english-wvq-morality \
  --split validation \
  --output results_siqa_val_us.json
 
# ── Base model on test ──
echo ""
echo "====== BASE MODEL - TEST ======"
python eval_social_iqa.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --split test \
  --output results_siqa_test_base.json
 
# ── US CultureLLM on test ──
echo ""
echo "====== US CULTURELLM - TEST ======"
python eval_social_iqa.py \
  --model $PROJECT_DIR/results/llama8b-english-wvq-morality \
  --split test \
  --output results_siqa_test_us.json
 
echo ""
echo "====== ALL DONE ======"
echo "Results saved to results_siqa_*.json"