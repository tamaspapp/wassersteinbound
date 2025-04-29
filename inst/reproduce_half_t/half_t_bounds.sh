#!/bin/bash

#SBATCH -J half_t_bounds
#SBATCH -c 2
#SBATCH --mem=50G
#SBATCH --output=./report/half_t_bounds%A_%a.out
#SBATCH --error=./report/half_t_bounds%A_%a.err

epsilons=(0.03 0.01 0.003 0.001 0.0003)

Rscript half_t_bounds.R ${epsilons[$SLURM_ARRAY_TASK_ID]}