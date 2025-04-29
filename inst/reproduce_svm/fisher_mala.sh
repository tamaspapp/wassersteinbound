#!/bin/bash

#SBATCH -J fisher-mala-svm
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -o fisher-mala-svm.out

srun Rscript fisher_mala.R