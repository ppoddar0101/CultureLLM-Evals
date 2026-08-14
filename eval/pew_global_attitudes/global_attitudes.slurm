#!/bin/bash
#SBATCH --job-name=global_attitudes_brazil
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
echo "TMPDIR = $TMPDIR"
echo "=========================================="

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd $PROJECT_DIR

python3 compute_global_attitudes_flash.py \
  --country "Italy" \
  --questions_json pew_questions.json \
  --ga_csv ga_master_italy_long_clean.csv \
  --norms_json cultural-norms.json \
  --chunk_size 4 --max_retries 4
  
