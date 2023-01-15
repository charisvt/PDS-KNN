#!/bin/bash
#SBATCH --partition=rome
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --cpus-per-task=4

module load gcc/10.2.0 mvapich2/2.3.4 openblas

mpicc -o knn knn.c -L/mnt/apps/aristotle/site/linux-centos7-x86_64/gcc-10.2.0/openblas-0.3.20-skoz3oyx4ekzpocbkfasvmt2oltlxh2z/lib -I/mnt/apps/aristotle/site/linux-centos7-x86_64/gcc-10.2.0/openblas-0.3.20-skoz3oyx4ekzpocbkfasvmt2oltlxh2z/include -lopenblas -fopenmp -O3

srun ./knn