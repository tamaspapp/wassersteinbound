#!/bin/bash

#SBATCH -J ula-svm
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o ula-svm.out

srun Rscript ula.R