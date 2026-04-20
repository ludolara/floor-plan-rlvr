#!/bin/sh
#SBATCH --job-name=nl-inference
#SBATCH --output=log_nl/job_output.log
#SBATCH --error=log_nl/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=32G 
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=6:00:00

export PYTHONPATH="$PYTHONPATH:/."

module load cuda/12
module load python/3.11
module load arrow gcc opencv

source $SCRATCH/env/vllm/bin/activate

# python src/pred/prompt_pred.py \
#     --num_examples 3

python src/pred/nl_pred.py 