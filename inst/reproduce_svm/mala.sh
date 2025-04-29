#!/bin/bash

#SBATCH -J mala-svm
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -o mala-svm.out

srun Rscript mala.R