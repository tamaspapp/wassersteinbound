library(wassersteinbound)
library(doParallel)

#########
# Setup #
#########
source("../helpers.R")

if(!file.exists("svmData.RData")){source("prelim.R")} # Ensure we run this first
load(file = "svmData.RData")
SetSeed_pcg32(seed)

ncores <- 16

h_mala <- 0.15 / t^(1/6)

mala_params <- list(
  "Sigma" = rep(h_mala^2, t),
  "thresh" = 0)

iter_mala <- 3e4
thin_mala <- 15
iters_mala <- seq(0,iter_mala,thin_mala)

#######################
# Single-chain output #
#######################
mala_out <- get_singlechain_output(mala_cpp, svm_params, mala_params, y0s_prior, iter_mala, thin_mala, ncores, seed)

# cl <- parallel::makeCluster(ncores)
# doParallel::registerDoParallel(cl)
# mala_out <- foreach(r = 1:R, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
#   SetSeed_pcg32(seed, r)
#   mcmc_out <- mala_cpp(svm_params, mala_params, y0_, iter_mala, thin_mala)
#   return(mcmc_out$xs)
# }
# parallel::stopCluster(cl)
# mala_out <- format_singlechain_out(mala_out)

##################
# Coupled output #
##################

cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)
mala_cpl_out <- foreach(r = 1:R, xinfinity_ = xinfinitys, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
  SetSeed_pcg32(seed, r)
  mcmc_out <- mala_twoscalecrn_cpp(svm_params, mala_params, xinfinity_, y0_, iter_mala, thin_mala)
  return(mcmc_out$squaredist)
}
parallel::stopCluster(cl)

##############
# Get bounds #
##############
# Empirical bound
emp_est_mala <- get_empirical_w2sq(mala_out, tail(mala_out,1)[[1]], ncores)
empirical_df_mala <- get_empirical_bounds(emp_est_mala, iters_mala, 
                                          debias = c(1e4, 2.5e4), conf_level = 0.95)
# Coupling bound
cpl_df <- get_coupling_bound(mala_cpl_out, iters_mala, 
                             name = "Coupling", conf_level = 0.95, boot_reps = 1e3)

# Output to plot
plt_df_mala <- rbind(empirical_df_mala, cpl_df)
save(plt_df_mala, file = "mala_plot.RData")
