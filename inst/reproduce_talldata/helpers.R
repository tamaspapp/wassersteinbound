library(doParallel)
library(Matrix)
library(wassersteinbound)
library(rstan)

#####
# Dataset importing and cleaning
#####
# Pima dataset
readPima <- function() {
  pima_data <- read.csv("../datasets/pima-indians-diabetes-new.csv", header = T)
  n <- nrow(pima_data)
  d <- ncol(pima_data) - 1
  
  y <- 2 * pima_data[, d+1] - 1
  
  X <- as.matrix(pima_data[, 1:d])
  X <- scale(X)    # Center and scale
  X <- cbind(1, X) # Add intercept
  
  yX <- sweep(X,1,y,"*")
  colnames(yX) <- NULL
  return(list("yX" = yX, "y" = y, "X" = X))
}

# DS1 dataset, tall data
readDS1 <- function() {
  ds1_data <- read.csv("../datasets/ds1.10.csv.bz2", header = F)
  n <- nrow(ds1_data)
  d <- ncol(ds1_data) - 1
  
  y <- 2 * ds1_data[, d+1] - 1
  
  X <- as.matrix(ds1_data[, 1:d])
  X <- scale(X)    # Center and scale
  X <- cbind(1, X) # Add intercept
  
  yX <- sweep(X,1,y,"*")
  return(list("yX" = yX, "y" = y, "X" = X))
}

#####
# Mode-finding routine
#####
find_mode_BFGS <- function(init, target_params, maxit_ = 100, reltol_ = 1e-18){
  # Set up target eval functions
  negLogPi     <- function(theta) {potential_cpp(target_params, theta)}
  negGradLogPi <- function(theta) {gradpotential_cpp(target_params, theta)}
  negHessLogPi <- function(theta) {hesspotential_cpp(target_params, theta)}
  
  # Find the mode with BFGS
  optim_out <- optim(init, negLogPi, gr = negGradLogPi,
                     control = list(maxit = maxit_, reltol = reltol_),
                     method = "BFGS")
  mode <- optim_out$par
  mode_evals <- list("mode" = mode,
                     "gradient_at_mode" = negGradLogPi(mode),
                     "hessian_at_mode" = negHessLogPi(mode))
  
  return(mode_evals)
}

#####
# Compile Stan model for ADVI
#####
logreg_stan <- stan_model("logreg_model.stan")

#####
# Parallel MCMC
#####
concatenate_within_list <- function(x, ...) Map(function(...) c(...), x, ...) # f(A, B) = list(c(A[[1]], B[[1]]), c(A[[2]], B[[2]]), ...)

run_singlechain_burnin <- function(sampler, target_params,
                                   sampler_params, run_params, x0s,
                                   ncores, seed, rng_stream_offset = 0) {
  R    <- run_params$reps
  iter <- run_params$iter
  thin <- iter
  
  cl <- parallel::makeCluster(ncores) # Sample chains in parallel
  doParallel::registerDoParallel(cl)
  
  burnin_out <-
    foreach(i = 1:R, x0 = x0s, .packages = c("wassersteinbound", "Matrix")) %dopar% {
      SetSeed_pcg32(seed, - R - rng_stream_offset - i)
      xs <- sampler(target_params, sampler_params, x0, iter, thin)$xs
      return(list(as.vector(tail(xs, 1))))
    }
  parallel::stopCluster(cl)
  
  return(burnin_out) # Output: list of (final sample)
}

run_singlechain <- function(sampler, target_params,
                            sampler_params, run_params, x0s,
                            ncores, seed, rng_stream_offset = 0) {
  
  x0s <- run_singlechain_burnin(sampler, target_params, sampler_params, run_params, x0s, ncores, seed, rng_stream_offset)
  
  R    <- run_params$reps
  iter <- run_params$iter
  thin <- run_params$thin
  
  cl <- parallel::makeCluster(ncores) # Sample chains in parallel
  doParallel::registerDoParallel(cl)
  
  singlechain_out <-
    foreach(i = 1:R, x0_ = x0s, .packages = c("wassersteinbound","Matrix")) %dopar% {
      SetSeed_pcg32(seed, R + i + rng_stream_offset)
      xs <- sampler(target_params, sampler_params, x0_, iter, thin)$xs
      return(list(xs))
    }
  parallel::stopCluster(cl)
  
  return(singlechain_out)
}

run_coupledchains_burnin <- function(coupled_sampler, 
                                     target1_params = NA, target2_params,  # Exact sampler is always the SECOND ONE
                                     sampler_params, run_params, x0s, y0s, # Exact samples in the Y-chain 
                                     ncores, seed, rng_stream_offset = 0) {
  
  R    <- run_params$reps
  iter <- run_params$burnin
  thin <- iter
  
  cl <- parallel::makeCluster(ncores) # Sample chains in parallel
  doParallel::registerDoParallel(cl)
  
  if(anyNA(target1_params)){
    burnin_out <- 
      foreach(i = 1:R, x0 = x0s, y0 = y0s, 
              .combine = concatenate_within_list, .multicombine = T,
              .packages = c("wassersteinbound","Matrix")) %dopar% {
                SetSeed_pcg32(seed, - R - rng_stream_offset + i)
                cpl_out <- coupled_sampler(target2_params, sampler_params, x0, y0, iter, thin)
                return(list("x0" = list(tail(cpl_out$xs, 1)), "y0" = list(tail(cpl_out$ys, 1))))
              }
  } else {
    burnin_out <- 
      foreach(i = 1:R, x0 = x0s, y0 = y0s, 
              .combine = concatenate_within_list, .multicombine = T,
              .packages = c("wassersteinbound","Matrix")) %dopar% {
                SetSeed_pcg32(seed, - R - rng_stream_offset + i)
                cpl_out <- coupled_sampler(target1_params, target2_params, sampler_params, x0, y0, iter, thin)
                return(list("x0" = list(tail(cpl_out$xs, 1)), "y0" = list(tail(cpl_out$ys, 1))))
              }
  }
  parallel::stopCluster(cl)
  
  return(burnin_out) # Output: list of (final pair of samples)
}

run_coupledchains <- function(coupled_sampler, 
                              target1_params = NA, target2_params,  # Exact sampler is always the SECOND ONE
                              sampler_params, run_params, x0s, y0s, # Exact samples in the Y-chain 
                              ncores, seed, rng_stream_offset = 0) {
  
  burnin_out <- run_coupledchains_burnin(coupled_sampler, target1_params, target2_params, sampler_params, run_params, x0s, y0s, ncores, seed, rng_stream_offset)
  
  x0s <- burnin_out$x0
  y0s <- burnin_out$y0
  
  R    <- run_params$reps
  iter <- run_params$iter
  thin <- run_params$thin
  
  # Sample chains in parallel
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  if(anyNA(target1_params)){
    coupling_out <- 
      foreach(i = 1:R, x0 = x0s, y0 = y0s, 
              .combine = concatenate_within_list, .multicombine = T,
              .packages = c("wassersteinbound","Matrix")) %dopar% {
                SetSeed_pcg32(seed, R + i + rng_stream_offset)
                cpl_out <- coupled_sampler(target2_params, sampler_params, x0, y0, iter, thin)
                return(list("xs" = list(cpl_out$xs), "ys" = list(cpl_out$ys), "squaredist" = list(cpl_out$squaredist)))
              }
  } else {
    coupling_out <- 
      foreach(i = 1:R, x0 = x0s, y0 = y0s, 
              .combine = concatenate_within_list,  .multicombine = T,
              .packages = c("wassersteinbound","Matrix")) %dopar% {
                SetSeed_pcg32(seed, R + i + rng_stream_offset)
                cpl_out <- coupled_sampler(target1_params, target2_params, sampler_params, x0, y0, iter, thin)
                return(list("xs" = list(cpl_out$xs), "ys" = list(cpl_out$ys), "squaredist" = list(cpl_out$squaredist)))
              }
  }
  parallel::stopCluster(cl)
  
  return(coupling_out)
}