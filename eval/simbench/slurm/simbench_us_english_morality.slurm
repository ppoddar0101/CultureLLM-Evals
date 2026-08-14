#!/bin/bash
#SBATCH --job-name=simbench_us_morality
#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate culturellm

export HF_CACHE_LOCAL=$HF_HOME
export LOCAL_WORKDIR=$PROJECT_DIR/eval/simbench
export OMP_NUM_THREADS=4

cd $PROJECT_DIR/eval/simbench

pip install google-generativeai

python -c "import google.generativeai; print('genai OK')" 2>&1
python -c "import openai; print('openai OK')" 2>&1

#python generate_answers.py \
#    --input_file SimBenchTargets_US_all.pkl \
#    --output_file results/us_english_morality_verb.pkl \
#    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
#    --adapter_path "$PROJECT_DIR/results/llama8b-english-wvq-morality" \
#    --method verbalized

#echo "=== Brazil ==="
#python generate_answers.py --input_file SimBenchTargets_Brazil_pop.pkl --output_file results/brazil_scoring/brazil_gemini_verb.pkl --model_name "gemini-2.5-flash" --method verbalized

#echo "=== India ==="
#python generate_answers.py --input_file SimBenchTargets_India_pop.pkl --output_file results/india_scoring/india_gemini_verb.pkl --model_name "gemini-2.5-flash" --method verbalized

#echo "=== Italy ==="
#python generate_answers.py --input_file SimBenchTargets_Italy_pop.pkl --output_file results/italy_scoring/italy_gemini_verb.pkl --model_name "gemini-2.5-flash" --method verbalized

#echo "=== France ==="
#python generate_answers.py --input_file SimBenchTargets_France_pop.pkl --output_file results/france_scoring/france_gemini_verb.pkl --model_name "gemini-2.5-flash" --method verbalized

echo "=== US ==="
python generate_answers.py --input_file SimBenchTargets_US_all.pkl --output_file results/us_scoring/us_gemini_verb.pkl --model_name "gemini-2.5-flash" --method verbalized

echo "=== All done ==="
