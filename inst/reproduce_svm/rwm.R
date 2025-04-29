library(wassersteinbound)
library(doParallel)

# 0. Setup ####
source("../helpers.R")
if(!file.exists("svmData.RData")){source("prelim.R")} # Ensure we run this first
load(file = "svmData.RData")
SetSeed_pcg32(seed)

ncores <- 16

rwm_experiment <- function(rwm_params, iter, thin, seed_ = seed, svm_params_ = svm_params){
  
  iters <- seq(0, iter, thin)
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  singlechain_out <- foreach(r = 1:R, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
    SetSeed_pcg32(seed_, r)
    mcmc_out <- rwm_cpp(svm_params_, rwm_params, y0_, iter, thin)
    return(mcmc_out$xs)
  }
  parallel::stopCluster(cl)
  singlechain_out <- format_singlechain_out(singlechain_out)
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  coupling_out <- foreach(r = 1:R, xinfinity_ = xinfinitys, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
    SetSeed_pcg32(seed_, r)
    mcmc_out <- rwm_twoscalegcrn_cpp(svm_params_, rwm_params, xinfinity_, y0_, iter, thin)
    return(mcmc_out$squaredist)
  }
  parallel::stopCluster(cl)
  
 return(list("singlechain_out" = singlechain_out, "coupling_out" = coupling_out)) 
}

get_gcrn_results <-  function(rwm_params, iter, thin, seed_ = seed, svm_params_ = svm_params){
  rwm_params$thresh <- 0
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  coupling_out <- foreach(r = 1:R, xinfinity_ = xinfinitys, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
    SetSeed_pcg32(seed_, r)
    mcmc_out <- rwm_twoscalegcrn_cpp(svm_params_, rwm_params, xinfinity_, y0_, iter, thin)
    return(mcmc_out$squaredist)
  }
  parallel::stopCluster(cl)
  
  return(coupling_out)
}

#####

# 1. Optimal step size ####
h_rwm_big <- 0.25 / t^(1/2)
rwm_params_big <- list(
  "Sigma" = rep(h_rwm_big^2, t),
  "thresh" = Inf)
iter_big <- 1.5e6
thin_big <- 5e2
iters_big <- seq(0,iter_big,thin_big)
debias_big <- c(5e5, 1.25e6)

# Acceptance rate: 24%
# rwm_cpp(svm_params, rwm_params_big, y0s_prior[[1]], iter_big, thin_big)$acc_rate_x

out_big <- rwm_experiment(rwm_params_big, iter_big, thin_big)

empirical_big <- get_empirical_w2sq(out_big$singlechain_out, tail(out_big$singlechain_out,1)[[1]], ncores)
empirical_df_big <- get_empirical_bounds(empirical_big, iters_big, debias_big, conf_level = 0.95)
coupling_df_big <- get_coupling_bound(out_big$coupling_out, iters_big, name = "Coupling", conf_level = 0.95, boot_reps = 1e3)
rwm_big <- rbind(empirical_df_big, coupling_df_big)

# Also do GCRN coupling
coupling_df_big <- get_coupling_bound(get_gcrn_results(rwm_params_big, iter_big, thin_big), iters_big, name = "Coupling", conf_level = 0.95, boot_reps = 1e3)
rwm_big_gcrn <- rbind(empirical_df_big, coupling_df_big)
#####

# 2. Small step size ####
h_rwm_small <- 0.1 / t^(1/2)
rwm_params_small <- list(
  "Sigma" = rep(h_rwm_small^2, t),
  "thresh" = Inf)
iter_small <- 1.5e6
thin_small <- 5e2
iters_small <- seq(0,iter_small,thin_small)
debias_small <- c(7e5, 1.25e6)

# Acceptance rate: 64%
# rwm_cpp(svm_params, rwm_params_small, y0s_prior[[1]], iter_small, thin_small)$acc_rate_x

out_small <- rwm_experiment(rwm_params_small, iter_small, thin_small)

empirical_small <- get_empirical_w2sq(out_small$singlechain_out, tail(out_small$singlechain_out,1)[[1]], ncores)
empirical_df_small <- get_empirical_bounds(empirical_small, iters_small, debias_small, conf_level = 0.95)
coupling_df_small <- get_coupling_bound(out_small$coupling_out, iters_small, name = "Coupling", conf_level = 0.95, boot_reps = 1e3)
rwm_small <- rbind(empirical_df_small, coupling_df_small)

# Also do GCRN coupling
coupling_df_small <- get_coupling_bound(get_gcrn_results(rwm_params_small, iter_small, thin_small), iters_small, name = "Coupling", conf_level = 0.95, boot_reps = 1e3)
rwm_small_gcrn <- rbind(empirical_df_small, coupling_df_small)
#####

save(rwm_big, rwm_small, 
     coupling_df_big, coupling_df_small, 
     empirical_big, empirical_small, 
     rwm_big_gcrn, rwm_small_gcrn, file = "rwm_plot.RData")
