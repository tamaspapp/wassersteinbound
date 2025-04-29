#!/bin/bash

JobID=$(sbatch --parsable half_t_mcmc_singlechain.sh)
echo "Job ID: ${JobID}"
echo ""

ArrayID=$(sbatch --parsable --array=0-4 half_t_mcmc_coupled.sh)
echo "Job-Array ID: ${ArrayID}"
echo ""

sbatch --array=0-4 --dependency=aftercorr:${ArrayID},${JobID} half_t_bounds.sh