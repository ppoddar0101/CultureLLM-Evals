#!/bin/bash
#SBATCH --job-name=culturebank
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=c05-05,c06-05
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

pip install openai --quiet

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

export OPENAI_API_KEY="$OPENAI_API_KEY"  # set this in your environment$OPENAI_API_KEY"  # set this in your environment
python culturebank_score.py --input outputs/culturebank/responses_flash_Italy_20260520_171011.csv