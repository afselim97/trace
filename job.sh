#!/bin/bash
#SBATCH --job-name=infer_from_sims
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=pi-jnovembre

# Load necessary modules
module load python
source activate main
cd data

# Get the list of files
FILES=(demes/*.yaml)

# Run the Python script for the current file
./run_sim_infer.sh "${FILES[$SLURM_ARRAY_TASK_ID]}"