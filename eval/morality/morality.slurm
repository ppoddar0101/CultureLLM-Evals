#!/bin/bash
#SBATCH --job-name=morality
#SBATCH --partition=nlp
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

#python compute_morality.py \
#  --checkpoint_dir models/culturellm-france-8b-morality \
#  --country "France" \
#  --gt_csv wvs_w7.csv \
#  --norms_json cultural-norms.json
  
python3 compute_morality_gemini.py \
  --country "France" \
  --gt_csv wvs_w7.csv \
  --norms_json cultural-norms.json \
  --out_dir ./morality_outputs \
  --temperature 0.7 --top_p 0.9 \
  --bootstrap 1000