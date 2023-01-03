#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=15
#SBATCH --nodes=3
#SBATCH --time=1:00:00
# module load gcc/9.2.0 openmpi/4.0.3

module load gcc openmpi

mpicc example.c -lblas -o example

srun ./example