#!/bin/bash
#SBATCH -J perplexity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb

# activating the environment
module load python
source venv/bin/activate

# Run code
srun python ./main.py