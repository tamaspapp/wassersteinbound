#!/bin/bash

#SBATCH -J bench
#SBATCH -c 32
#SBATCH --mem-per-cpu=14G
#SBATCH -o bench.out

srun Rscript benchmark.R