#!/bin/bash
#SBATCH --job-name=jcdt4
#SBATCH --account=ms
#SBATCH --output=output/%A_res.out
#SBATCH --error=output/%A_err.out
#SBATCH --partition=node2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 30-00:00:00

# Job Execution Script
echo "Starting job on $(hostname)"

# Ensure we are in the correct project directory
cd /home/gpuadmin/cssin/cond_Jacobian
echo "Current working directory: $(pwd)"

# Verify the Python environment being used
echo "Using Python executable from:"
uv run which python

# Update dependencies
uv sync

# Run the python script
echo "Running script..."
uv run jacobian_classification_diff_timestep.py --num_projections 1 \
                                                --seed 36 \
                                                --timestep_indices 0

echo "Job finished"
exit 0