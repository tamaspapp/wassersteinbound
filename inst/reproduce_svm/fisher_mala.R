library(wassersteinbound)
library(doParallel)

# Problem parameters and helper functions
source("../helpers.R")
if(!file.exists("svmData.RData")){source("prelim.R")} # Ensure we run this first
load(file = "svmData.RData")

ncores <- 16

# MCMC parameters ####
h_mala <- 0.15 / t^(1/6)

fisher_mala_params <- list(
  "sigma0" = h_mala,
  "acceptance_rate_target" = 0.574,
  "learning_rate" = 0.015,
  "damping_factor" = 10)

iter_fmala <- 1.25e4
thin_fmala <- 5
iters_fmala <- seq(0,iter_fmala,thin_fmala)

# Run MCMC ####
fmala_out <- get_singlechain_output(fisher_mala_cpp, svm_params, fisher_mala_params, y0s_prior, iter_fmala, thin_fmala, ncores, seed)

# Get bound ####
emp_est_fmala <- get_empirical_w2sq(fmala_out, tail(fmala_out, 1)[[1]], ncores)
empirical_df_fmala <- get_empirical_bounds(emp_est_fmala, iters_fmala, 
                                           debias = c(7.5e3, 9e3), conf_level = 0.95)

# Output to plot ####
plt_df_fmala <- empirical_df_fmala
save(plt_df_fmala, file = "fmala_plot.RData")
