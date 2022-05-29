#!/bin/bash

#SBATCH --output=results/out.txt
#SBATCH --error=results/err.txt
#SBATCH --nodes=2          # number of nodes
#SBATCH --ntasks=2         # number of processor cores (i.e. tasks)
#SBATCH --tasks-per-node=1 # number of tasks per node
#SBATCH --exclusive
#SBATCH --time=00:10:00    # walltime

module load miniconda

srun python pool.py
