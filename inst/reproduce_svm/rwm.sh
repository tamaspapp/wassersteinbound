#!/bin/bash

#SBATCH -J rwm-svm
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -o rwm-svm.out

srun Rscript rwm.R