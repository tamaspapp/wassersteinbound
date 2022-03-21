
# Run MALA (optimal scaling), plot bounds
source("mala_run.R")

# Run RWM (optimal scaling), plot bounds
source("rwm_run.R")

# Run RWM (small scaling), plot bounds
source("rwm_run_smallscaling.R")

####
# Supplement
####
# Run MALA from overdispersed start, plot bounds
# (script depends on output of mala_run.R)
source("mala_run_from2stationary.R")