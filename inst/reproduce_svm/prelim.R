library(wassersteinbound)
library(doParallel)

# RNG and parallel computing
seed <- 12345 
set.seed(seed)

ncores <- 4

# Helpers
SampleLatentVariables <- function(t, sig, phi) {
  x <- rep(NA, t)
  
  x[1] <- rnorm(1, mean = 0, sd = sig / sqrt(1 - phi^2))
  
  if(t > 1){
    for(i in 2:t){
      x[i] <- rnorm(1, mean = phi * x[i-1], sd = sig)
    }
  }
  return(x)
}

# Model parameters
t <- 360
beta <- 0.65
sig  <- 0.15
phi  <- 0.98

y <- rnorm(t, 0 , beta * exp(0.5 * SampleLatentVariables(t, sig, phi))) # Sample the data from the model

svm_params <- list("y" = y,
                   "beta" = beta, 
                   "sigma" = sig, 
                   "phi" = phi,
                   "target_type" = "stochastic_volatility")

# MCMC parameters
R <- 1024

# Sample from the prior = initial distribution
y0s_prior <- lapply(1:R, function(x){SampleLatentVariables(t, sig, phi)})

# Sample from the target by long MCMC runs
mala_params_prelim <- list("Sigma" = rep(0.15 / t^(1/6), t)^2)
iter_prelim <- 1e4 # Sufficient for convergence

cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)
xinfinitys <- foreach(r = 1:R, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
  SetSeed_pcg32(seed, r - R - 1)
  mcmc_out <- mala_cpp(svm_params, mala_params_prelim, y0_, iter_prelim, iter_prelim)
  return(as.vector(tail(mcmc_out$xs, 1)))
}
parallel::stopCluster(cl)

save(seed, ncores, 
     R, xinfinitys, y0s_prior,
     t, svm_params, file = "svmData.RData")
