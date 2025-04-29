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
stationary_samples <- foreach(param = problem_params) %do% {
  replicate(R, param$mu + as.vector(rnorm(param$d) %*% param$Sigma_ULA_chol_u), simplify = F)
}

###
# 1. Coupling-based estimators
###
cpl_iter <- 1000
cpl_thin <- 1

cpl_run <- function(cpl_iter, cpl_thin, seed, ncores) {
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  cpl_iters <- seq(0, cpl_iter, cpl_thin)
  
  cpl_out <-
    foreach(params_ = problem_params, x_0s = init_samples, x_inftys = stationary_samples, .combine = "rbind")   %:%
    foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %do% {
      
      target_params <- params_
      sampler_params <- list("gamma" = gamma_, "delta" = params_$h, "Sigma" = rep(1, params_$d))
      
      squaredists <- 
        foreach(i = 1:R, x_0 = x_0s, x_infty = x_inftys, .combine = "rbind", .packages = c("wassersteinbound", "Matrix")) %dopar% {
          SetSeed_pcg32(seed, i); obab_CRN_cpp(target_params, sampler_params, x_infty, x_0, cpl_iter, cpl_thin)$squaredist
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
    foreach(params_ = problem_params, x_0s = init_samples, x_inftys = stationary_samples, .combine = "rbind") %:%
    foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %do% {
      
      target_params <- params_
      sampler_params <- list("gamma" = gamma_, "delta" = params_$h, "Sigma" = rep(1, params_$d))
      
      # rbind_within_list <- function(x, ...) Map(function(...) rbind(...), x, ...)
      rbind_within_list <- function(x, ...) Map(function(...) do.call("rbind", list(...)), x, ...)
      
      # Memory management
      suppressWarnings(rm(xs))
      xs <- foreach(i = 1:R, x_0 = x_0s, 
                    .combine = rbind_within_list, .init = vector("list", length(empirical_iters)), 
                    .multicombine = T, 
                    .packages = c("wassersteinbound", "Matrix")) %dopar% {
       SetSeed_pcg32(seed, i); asplit(obab_cpp(target_params, sampler_params, x_0, empirical_iter, empirical_thin)$xs, 1)
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

empirical_upper <- get_empirical_w2sq_upper(empirical_out, 300, empirical_iter)
empirical_lower <- get_empirical_w2sq_lower(empirical_out, 300, empirical_iter)

###
# 3. Exact distance
###
exact_iter <- 50
exact_thin <- 1

exact_out <-
  foreach(params_ = problem_params, .combine = "rbind") %:%
  foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %do% {
      Sigma_0_diag <- diag(params_$Sigma_0)
      Sigma_target_diag <- eigen(params_$Sigma,T,only.values = T)$values
      exact_w2sq <- get_obab_exact_w2sq(exact_iter, exact_thin, params_$h, gamma_, 
                                        params_$mu_0, Sigma_0_diag, params_$mu, Sigma_target_diag)
      return(data.frame(exact_w2sq, "gamma" = gamma_label, "d" = params_$d))
    }

###
# 4. Calculate the mixing times
###
w2sq_thresh <- 6

mixing_times_unadjusted <-
  rbind(
    cbind(get_mixing_times(cpl_out, w2sq_thresh), "Estimator" = "Coupling"),
    cbind(get_mixing_times(empirical_upper, w2sq_thresh), "Estimator" = "U"),
    cbind(get_mixing_times(empirical_lower, w2sq_thresh), "Estimator" = "L"),
    cbind(get_mixing_times(exact_out, w2sq_thresh), "Estimator" = "Exact")
  )

# save(mixing_times_unadjusted, file = "convergence_ar1_unadjusted.RData")
