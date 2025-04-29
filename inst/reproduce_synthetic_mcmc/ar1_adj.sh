#!/bin/bash

#SBATCH --job-name ar1-convergence-adj
#SBATCH --output adj-%j.out
#SBATCH --mem-per-cpu=16gb
#SBATCH -c 8

srun Rscript convergence_ar1_adjusted.R
