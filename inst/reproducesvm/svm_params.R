#####
# Problem setup

library(wassersteinbound)
library(doParallel)


# Set seed for RNGs
seed <- 12345
set.seed(seed)
SetSeed_cpp(seed)

# Number of cores to be used
ncores <- parallel::detectCores()



# Model parameters
t <- 360L # dimension of the problem

beta <- 0.65
sig  <- 0.15
phi  <- 0.98

# Sample the data from the model
y <- beta * rnorm(t) * exp(0.5 * SampleLatentVariables(t, sig, phi))



# Number of MCMC chains ("replicates")
R <- 1000L
