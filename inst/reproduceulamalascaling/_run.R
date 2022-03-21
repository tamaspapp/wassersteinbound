
# MALA: Run MCMC and compute mixing time bounds
source("mala_run.R")

# ULA: Run MCMC and compute mixing time bounds and exact mixing times
source("ula_run.R")
source("ula_exact_w2sq.R")

# Produce mixing time bound plot
source("mala_ula_joint_plot.R")

####
# Supplementary material:
####
# Bounds on + exact W2sq(target, stationary) for ULA
source("ula_target_bias.R")
