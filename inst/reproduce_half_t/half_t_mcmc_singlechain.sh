#!/bin/bash

#SBATCH -J half_t_mcmc
#SBATCH -c 25
#SBATCH --mem=50G
#SBATCH --output=./report/half_t_mcmc_singlechain.out
#SBATCH --error=./report/half_t_mcmc_singlechain.err

Rscript half_t_mcmc_singlechain_exact.R 25