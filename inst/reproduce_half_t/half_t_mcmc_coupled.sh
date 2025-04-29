#!/bin/bash

#SBATCH -J half_t_mcmc_coupled
#SBATCH -c 20
#SBATCH --mem=40G
#SBATCH --output=./report/half_t_mcmc_coupled%A_%a.out
#SBATCH --error=./report/half_t_mcmc_coupled%A_%a.err

epsilons=(0.03 0.01 0.003 0.001 0.0003)

Rscript half_t_mcmc_coupled.R ${epsilons[$SLURM_ARRAY_TASK_ID]} $(($SLURM_CPUS_PER_TASK))