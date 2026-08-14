#!/bin/bash
#SBATCH --job-name=ipip_verb_8b_english
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=c05-05,c06-05
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

#python personality_eval_verbalize.py --country "France"
python personality_eval_gemini.py --country "Italy"