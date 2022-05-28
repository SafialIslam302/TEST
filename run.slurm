#!/bin/bash
#
#SBATCH --job-name=test_mpi
#SBATCH --output=res_mpi.txt
#
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

module load OpenMPI
module load openmpi
mpicc wiki_mpi_example.c -o hello.mpi
