library(wassersteinbound)
library(Matrix)
library(doParallel)
library(dplyr)

source("helpers_ar1.R")

ncores <- 8 # Parallel computing
R <- 1024 # Number of chains

seed <- 12345 # Set up RNG
set.seed(seed)

###
# 0. Setup
###
rho <- 0.5 # Target parameters: autocorrelation and dimension
ds <- rev(c(128, 256, 512, 1024))
hs <- ds^(-1/6) # Step sizes

# Collate parameters associated to each dimension, including friction parameters
problem_params <- get_problem_params(ds, hs)

# Samples from the initial distribution
init_samples <- foreach(param = problem_params) %do% {
  replicate(R, param$mu_0 + as.vector(rnorm(param$d) %*% param$Sigma_0_chol_u), simplify = F)
}

# Samples from the stationary distribution
target_samples <- foreach(param = problem_params) %do% {
  replicate(R, param$mu + as.vector(rnorm(param$d) %*% param$Sigma_chol_u), simplify = F)
}

###
# 1. Coupling-based estimators
###
cpl_iter <- 3000
cpl_thin <- 1

cpl_run <- function(cpl_iter, cpl_thin, seed, ncores) {
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  cpl_iters <- seq(0, cpl_iter, cpl_thin)
  
  cpl_out <-
    foreach(params_ = problem_params, x_0s = init_samples, x_inftys = target_samples, .combine = "rbind")   %:%
    foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %do% {
      
      target_params <- params_
      sampler_params <- list("gamma" = gamma_, "delta" = params_$h, "Sigma" = rep(1, params_$d))
      
      squaredists <- 
        foreach(i = 1:R, x_0 = x_0s, x_infty = x_inftys, .combine = "rbind", .packages = c("wassersteinbound", "Matrix")) %dopar% {
          SetSeed_pcg32(seed, i); horowitz_CRN_cpp(target_params, sampler_params, x_infty, x_0, cpl_iter, cpl_thin)$squaredist
        }
      
      data.frame("w2sq" = colMeans(squaredists), "iter" = cpl_iters, "gamma" = gamma_label, "d" = params_$d)
    }
  parallel::stopCluster(cl)
  
  return(cpl_out)
} 
cpl_out <- cpl_run(cpl_iter, cpl_thin, seed, ncores)

###
# 2. Empirical estimators
###
empirical_iter <- 1000
empirical_thin <- 1

empirical_run <- function(empirical_iter, empirical_thin, seed, ncores) {
 
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
   
  get_empirical_w2sq <- function(mcmc_out, x_ref) {
    fix_zeros <- function(x) ifelse(x < 0, 0, x) # Deal with values that are exactly zero
    foreach(x = mcmc_out, .packages = c("wassersteinbound"), .combine = "c") %dopar% {fix_zeros(w2sq_empirical(x, x_ref)$w2sq)}
  }

  empirical_iters <- seq(0, empirical_iter, empirical_thin)
  
  empirical_out <-
    foreach(params_ = problem_params, x_0s = init_samples, x_inftys = target_samples, .combine = "rbind") %:%
    foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %do% {
      
      target_params <- params_
      sampler_params <- list("gamma" = gamma_, "delta" = params_$h, "Sigma" = rep(1, params_$d))
  
      # rbind_within_list <- function(x, ...) Map(function(...) rbind(...), x, ...)
      rbind_within_list <- function(x, ...) Map(function(...) do.call("rbind", list(...)), x, ...)
      
      # Memory management
      suppressWarnings(rm(xs))
      xs <- foreach(i = 1:R, x_0 = x_0s, 
                    .combine = rbind_within_list, .init = vector("list", length(empirical_iters)), 
                    .multicombine = T, .maxcombine = 32,
                    .packages = c("wassersteinbound", "Matrix")) %dopar% {
        SetSeed_pcg32(seed, i); asplit(horowitz_cpp(target_params, sampler_params, x_0, empirical_iter, empirical_thin)$xs, 1)
      }
      x_ref <- do.call(rbind, x_inftys) # Independent samples from the target
      
      data.frame("w2sq" = get_empirical_w2sq(xs, x_ref), "iter" = empirical_iters, "gamma" = gamma_label, "d" = params_$d)
    }
  
  parallel::stopCluster(cl)
  return(empirical_out)
}
empirical_out <- empirical_run(empirical_iter, empirical_thin, seed, ncores)

# library(ggplot2)
# ggplot(empirical_out, aes(x = iter, y = w2sq))+
#   geom_line() +
#   facet_grid(d~gamma, scales = "free_y") +
#   scale_y_log10()

empirical_upper <- get_empirical_w2sq_upper(empirical_out, 400, empirical_iter)
empirical_lower <- get_empirical_w2sq_lower(empirical_out, 400, empirical_iter)


###
# 3. Exact distance
###
exact_iter <- 200
exact_thin <- 1

# Approximate exact Wasserstein distance
exact_run <- function(iter, thin, seed, ncores) {
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  # Assume that the target and the marginal distributions are simultaneously diagonalizable mean-zero Gaussians
  get_exact_w2sq_approximation <- function(xt, Sigma){
    eigen_Sigma <- eigen(Sigma)
    D <- eigen_Sigma$values
    U <- eigen_Sigma$vectors
    
    get_sd_in_principal_components <- function(X) apply(X %*% U, 2, sd) # Since t(U) %*% X[i,] is in the right basis
    
    only_positive <- function(x) x[x>0] # Extra tweak: we know we're overdispersed, so keep only positive values
    foreach(x = xt, .packages = c("wassersteinbound"), .combine = "c") %dopar% {sum(only_positive(get_sd_in_principal_components(x) - sqrt(D))^2)}
  }
  
  iters <- seq(0, iter, thin)
  
  exact_out <-
    foreach(params_ = problem_params, x_0s = init_samples, x_inftys = target_samples, .combine = "rbind") %:%
    foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %do% {
      
      target_params <- params_
      sampler_params <- list("gamma" = gamma_, "delta" = params_$h, "Sigma" = rep(1, params_$d))
      
      rbind_within_list <- function(x, ...) Map(function(...) rbind(...), x, ...)
      
      xs <- foreach(i = 1:R, x_0 = x_0s, 
                    .combine = rbind_within_list, .init = vector("list", length(iters)), 
                    .multicombine = T,.packages = c("wassersteinbound", "Matrix")) %dopar% {
                      SetSeed_pcg32(seed, i); asplit(horowitz_cpp(target_params, sampler_params, x_0, iter, thin)$xs, 1)
                    }
      
      data.frame("w2sq" = get_exact_w2sq_approximation(xs, params_$Sigma), 
                 "iter" = iters, "gamma" = gamma_label, "d" = params_$d)
    }
  
  parallel::stopCluster(cl)
  return(exact_out)
}
exact_out <- exact_run(exact_iter, exact_thin, seed, ncores)

###
# 4. Calculate the mixing times
###
w2sq_thresh <- 6

mixing_times_adjusted <-
  rbind(
    cbind(get_mixing_times(cpl_out, w2sq_thresh), "Estimator" = "Coupling"),
    cbind(get_mixing_times(empirical_upper, w2sq_thresh), "Estimator" = "U"),
    cbind(get_mixing_times(empirical_lower, w2sq_thresh), "Estimator" = "L"),
    cbind(get_mixing_times(exact_out, w2sq_thresh), "Estimator" = "Exact")
  )

save(mixing_times_adjusted, file = "convergence_ar1_adjusted.RData")
