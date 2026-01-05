#!/bin/bash
#
#SBATCH --job-name=jupyter
#SBATCH --account=ms
#SBATCH --output=output/%A_%a_res.txt
#SBATCH --error=output/%A_%a_err.txt
#SBATCH --partition=node1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH -t 12:00:00

hostname

uv sync
uv run --with jupyter jupyter notebook --notebook-dir=/home/gpuadmin/cssin/cond_Jacobian --ip=0.0.0.0 --port=35426 --no-browser --NotebookApp.token='f12ea7470e52b4b7724d003c0fb9525f671047fd865b6d13'

exit 0
