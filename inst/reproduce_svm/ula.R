library(wassersteinbound)
library(doParallel)
library(Matrix)

#########
# Setup #
#########
source("../helpers.R")

if(!file.exists("svmData.RData")){source("prelim.R")} # Ensure we run this first
load(file = "svmData.RData")
SetSeed_pcg32(seed)

ncores <- 4

h_mala <- 0.15 / t^(1/6)

mala_params <- list(
  "Sigma" = rep(h_mala^2, t),
  "thresh" = 0)

iter_mala <- 3e4
thin_mala <- 15
iters_mala <- seq(0,iter_mala,thin_mala)

# Laplace approximation
find_mode_BFGS <- function(init, target_params, maxit_ = 100, reltol_ = 1e-18){
  # Set up target eval functions
  negLogPi     <- function(theta) {potential_cpp(target_params, theta)}
  negGradLogPi <- function(theta) {gradpotential_cpp(target_params, theta)}
  negHessLogPi <- function(theta) {hesspotential_cpp(target_params, theta)}
  
  # Find the mode with BFGS
  optim_out <- optim(init, negLogPi, gr = negGradLogPi,
                     control = list(maxit = maxit_, reltol = reltol_),
                     method = "BFGS",
                     hessian = T)
  mode_evals <- list("mode" = optim_out$par,
                     "gradient_at_mode" = negGradLogPi(optim_out$par),
                     "hessian_at_mode" = optim_out$hess)
  
  return(mode_evals)
}
mode_svm <- find_mode_BFGS(rnorm(t), svm_params, 1e3)
laplace_params <- list("target_type" = "sparse_gaussian",
                       "mu"  = mode_svm$mode,
                       "Omega"  = as(mode_svm$hess,"dgCMatrix"),
                       "Omega_chol_u"  = as(chol(mode_svm$hess),"dgCMatrix"))

#######################
# Single-chain output #
#######################
ula_out <- get_singlechain_output(ula_cpp, laplace_params, mala_params, y0s_prior, iter_mala, thin_mala, ncores, seed)

##################
# Coupled output #
##################
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)
ula_cpl_out <- foreach(r = 1:R, xinfinity_ = xinfinitys, y0_ = y0s_prior, .packages = "wassersteinbound") %dopar% {
  SetSeed_pcg32(seed, r)
  mcmc_out <- ula_twoscalecrn_cpp(laplace_params, mala_params, xinfinity_, y0_, iter_mala, thin_mala)
  return(mcmc_out$squaredist)
}
parallel::stopCluster(cl)

##############
# Get bounds #
##############
# Empirical bound
emp_est_ula <- get_empirical_w2sq(ula_out, tail(ula_out,1)[[1]], ncores)
empirical_df_ula <- get_empirical_bounds(emp_est_ula, iters_mala, 
                                          debias = c(1e4, 2.5e4), conf_level = 0.95)
# Coupling bound
cpl_df <- get_coupling_bound(ula_cpl_out, iters_mala, 
                             name = "Coupling", conf_level = 0.95, boot_reps = 1e3)

# Exact squared distance
get_exact_w2sq_ula <- function(iter, thin, h, svm_params, laplace_params) {
  # Target parameters
  t <- length(svm_params$y)
  phi <- svm_params$phi
  sigma <- svm_params$sigma
  # Laplace parameters
  Omega <- as.matrix(laplace_params$Omega)
  
  # Start and end distribution
  mu_0 <- rep(0, t)
  Sigma_0 <- (sigma^2 / (1 - phi^2)) * phi^outer(1:t, 1:t, FUN = function(i, j) abs(i - j))
  mu_infty <- laplace_params$mu
  Sigma_infty <- solve((diag(t) - 0.25 * h^2 * Omega) %*% Omega)
  # "Slope matrix" in autoregression
  B_ula <- diag(t) - 0.5 * h^2 * Omega 
  
  # Get Wasserstein distance and output
  out_df <- data.frame("iter" = seq(0, iter, thin), 
                       "w2sq" = w2sq_convergence_gaussian_recursive_cpp(mu_0, Sigma_0, mu_infty, Sigma_infty, B_ula, nrow(B_ula), iter, thin, tol = 1e-3)$w2sq)
  out_df$ci_up <- out_df$ci_lo <- NA
  out_df$estimator <- "Exact"
  return(out_df)
}
exact_df <- get_exact_w2sq_ula(1e4, thin_mala, h_mala, svm_params, laplace_params)

# Output to plot
plt_df_ula <- rbind(empirical_df_ula, cpl_df, exact_df)
save(plt_df_ula, file = "ula_plot.RData")
