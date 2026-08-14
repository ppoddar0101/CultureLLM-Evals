#!/bin/bash
#SBATCH --job-name=prb_gemini
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm
pip install -q google-genai datasets

export OMP_NUM_THREADS=4
cd $PROJECT_DIR
mkdir -p prb_results

SUBSETS=(
    "Art_and_Entertainment"
    "Lifestyle_and_Personal_Development"
    "Society_and_Culture"
)
SEEDS=(42 123 456)

for SUBSET in "${SUBSETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running: ${SUBSET} / seed=${SEED}"
        python pairwise_gemini.py \
            --model_name gemini-2.5-flash \
            --subset "${SUBSET}" \
            --seed "${SEED}" \
            --output_dir prb_results
        echo "Sleeping 10s between runs..."
        sleep 10
    done
done

echo "All done."