source("helpers.R")
source("../bias_helpers.R")
library(dplyr)

# The plug-in estimators are ~not~ sensitive to the thinning factor: 
# thin_plugin = 1 also works fine.

pima_parameters <- list(
  "dataset" = "pima",
  "R" = 32,          # Replicates for parallel MCMC
  "iter" = 5e2,      # MCMC iterations after burn-in
  "burnin" = 1e2,    # Burn-in for MCMC
  "thin_plugin" = 5, # Extra thinning for plug-in estimators
  "ci_level" = 0.95, # Confidence interval for plot
  "ncores" = 4,
  "seed" = 12345
)

ds1_parameters <- list(
  "dataset" = "DS1",
  "R" = 32, 
  "iter" = 2e3, 
  "burnin" = 1e2,
  "thin_plugin" = 10,
  "ci_level" = 0.95,
  "ncores" = 4,
  "seed" = 12345
)

run_logistic_experiment <- function(parameters){
  
  seed <- parameters$seed
  set.seed(seed)

  ###############
  # Precomputation: import dataset, find mode, set up Laplace approximation
  ###############
  # Import dataset and set up true target parameters
  if(parameters$dataset == "pima"){
    logistic_params <- readPima()
  } else if (parameters$dataset == "DS1"){
    logistic_params <- readDS1()
  }
  n <- dim(logistic_params$yX)[1]
  d <- dim(logistic_params$yX)[2]
  logistic_params$target_type <- "logistic_regression"
  logistic_params$lambda <- rep(5, d) # Prior is a product of Gaussians
  
  # Find mode and get Laplace approximation
  mode_evals <- find_mode_BFGS(rnorm(d), logistic_params)
  
  mode <- mode_evals$mode
  gradient_at_mode <- mode_evals$gradient_at_mode
  hessian_at_mode <- mode_evals$hessian_at_mode
  
  laplace_params <- list("mu" = mode, # mean
                         "Omega" = hessian_at_mode, # precision matrix
                         "Sigma" = solve(hessian_at_mode),
                         "target_type" = "gaussian")
  
  # ADVI
  data_stan <- list('n' = n, 'dimension' = d, 
                    'x' = logistic_params$X, 'y' = (logistic_params$y+1)/2, 
                    "prior_mean" = rep(0,d), "prior_cov" = diag(rep(25,d)))
  mf_ADVI <- vb(logreg_stan, data = data_stan, 
                output_samples = 5e4, seed = seed, tol_rel_obj = 1e-5, iter = 5e4,
                "algorithm" = "fullrank")
  advi_output <- extract(mf_ADVI)
  
  advi_params <- list("mu" = colMeans(advi_output$beta),
                      "Sigma" = cov(advi_output$beta), #"Sigma" = diag(apply(advi_output$beta,2,var)),
                      "target_type" = "gaussian")
  advi_params$Omega <- solve(advi_params$Sigma)
  
  ###############
  # Sampler setup
  ###############
  ncores <- parameters$ncores
  R <- parameters$R 
  burnin <- parameters$burnin
  iter <- parameters$iter-1 # Iterations AFTER burn-in
  thin <- 1 # No thinning for MCMC output
  
  # Proposal generation
  h <- 1 / d^(1/6) # This is a small step size, yet the rejections impair the coupling bound
  Sigma_prop <- h^2 * laplace_params$Sigma
  
  # Laplace approximation
  laplace_mcmc_params <- list("Sigma" = Sigma_prop)
  laplace_coupled_run_params <-
    list("reps" = R,
         "burnin" = burnin,
         "iter" = iter, 
         "thin" = thin)
  
  # SGLD
  sgld_fraction <- 0.1 # Fraction of samples to use for batch size
  sgld_params <- laplace_mcmc_params
  sgld_params$"batch_size" <- ceiling(sgld_fraction * n)
  sgld_coupled_run_params <- laplace_coupled_run_params
  
  # SGLDcv
  sgldcv_fraction <- 0.01 # Fraction of samples to use for batch size
  sgldcv_params <- laplace_mcmc_params
  sgldcv_params$"batch_size" = ceiling(sgldcv_fraction * n)
  sgldcv_params$"mode" = mode
  sgldcv_params$"gradient_at_mode" = gradient_at_mode
  sgldcv_coupled_run_params <- laplace_coupled_run_params
  
  # ADVI
  advi_mcmc_params <- laplace_mcmc_params
  advi_coupled_run_params <- laplace_coupled_run_params
  
  #####
  # Parameters for plug-in bounds
  #####
  thinning_factor <- parameters$thin_plugin 
  samples_per_chain <- 1 + iter%/%thinning_factor
  sample_size <- (R/2) * samples_per_chain; print(paste("Sample size for proposed estimators:", sample_size))
  aggregate <- function(x) aggregate_mcmc_samples(x, thinning_factor)
  
  ###############
  # Sample everything
  ###############
  # Use different streams across algorithms: MALA samples are independent across runs
  
  # Initial samples for single-chain and for coupled MCMC
  x0 <- replicate(R, mode, simplify = F) # Starting at the mode because MALA can have issues in the tails
  y0 <- replicate(R, mode, simplify = F)
  
  laplace_out <- run_coupledchains(mala_CRN_2targets_cpp, laplace_params, logistic_params, laplace_mcmc_params, laplace_coupled_run_params, x0, y0, ncores, seed)
  sgld_out    <- run_coupledchains(sgld_mala_CRN_cpp, NA, logistic_params, sgld_params, sgld_coupled_run_params, x0, y0, ncores, seed)
  sgldcv_out  <- run_coupledchains(sgldcv_mala_CRN_cpp, NA, logistic_params, sgldcv_params, sgldcv_coupled_run_params, x0, y0, ncores, seed)
  advi_out    <- run_coupledchains(mala_CRN_2targets_cpp, advi_params, logistic_params, advi_mcmc_params, advi_coupled_run_params, x0, y0, ncores, seed)

  ###############
  # Bounds
  ###############
  # Get empirical bounds, using hedging, sample splitting, and variance reduction by coupling
  get_empirical_samplesplit <- function(coupled_output) {
    no_chains <- R
    half1 <- 1:(no_chains/2); half2 <- (no_chains/2+1):no_chains
    # We chose the thinning so that the samples were roughly i.i.d., so we
    # estimate the variance as if they were i.i.d.
    empirical <- get_empirical_bound(aggregate(coupled_output$xs[half1]), aggregate(coupled_output$ys[half1]),
                                     aggregate(coupled_output$xs[half2]), aggregate(coupled_output$ys[half2]), num_blocks = sample_size)
    
    return(empirical)
  }
  
  sgld_empirical    <- get_empirical_samplesplit(sgld_out)
  sgldcv_empirical  <- get_empirical_samplesplit(sgldcv_out)
  laplace_empirical <- get_empirical_samplesplit(laplace_out)
  advi_empirical    <- get_empirical_samplesplit(advi_out)
  
  get_var_decrease <- function(df) {
    df %>% group_by(Estimator) %>% mutate(var_improvement = (stderr_nocoupling / stderr)^2, .keep = "none")
  }
  var_improvement <- rbind(
    cbind(get_var_decrease(sgld_empirical), "algorithm" = "SGLD"),
    cbind(get_var_decrease(sgldcv_empirical), "algorithm" = "SGLD-cv"),
    cbind(get_var_decrease(laplace_empirical), "algorithm" = "Laplace"),
    cbind(get_var_decrease(advi_empirical), "algorithm" = "Full-rank VI")
  )
  print("Estimated factor of decrease in variance due to coupling:")
  print(var_improvement)
  
  sgld_cub    <- get_cub(sgld_out)
  sgldcv_cub  <- get_cub(sgldcv_out)
  laplace_cub <- get_cub(laplace_out)
  advi_cub    <- get_cub(advi_out)

  sgld_lb    <- get_lowerbound(sgld_out)
  sgldcv_lb  <- get_lowerbound(sgldcv_out)
  laplace_lb <- get_lowerbound(laplace_out)
  advi_lb    <- get_lowerbound(advi_out)
  
  # Collate
  estim_sgld    <- bind_rows(sgld_empirical, sgld_lb, sgld_cub)
  estim_sgldcv  <- bind_rows(sgldcv_empirical, sgldcv_lb, sgldcv_cub)
  estim_laplace <- bind_rows(laplace_empirical, laplace_lb, laplace_cub)
  estim_advi    <- bind_rows(advi_empirical, advi_lb, advi_cub)
  
  # Correct the bias of the Gelbrich lower bound
  correct_bias <- function(estim){
    estim$w2sq <- ifelse(!is.na(estim$bias), estim$w2sq - estim$bias, estim$w2sq)
    estim$bias <- NULL
    return(estim) 
  }
  estim_sgld <- correct_bias(estim_sgld)
  estim_sgldcv <- correct_bias(estim_sgldcv)
  estim_laplace <- correct_bias(estim_laplace)
  estim_advi <- correct_bias(estim_advi)
  
  ######
  # Convert to confidence intervals
  ci_level <- parameters$ci_level
  
  plt_df <- rbind(
    cbind(convert_and_get_ci(estim_sgld, z = qnorm(0.5 + 0.5 * ci_level)), "algorithm" = "SGLD"),
    cbind(convert_and_get_ci(estim_sgldcv, z = qnorm(0.5 + 0.5 * ci_level)), "algorithm" = "SGLD-cv"),
    cbind(convert_and_get_ci(estim_laplace, z = qnorm(0.5 + 0.5 * ci_level)), "algorithm" = "Laplace"),
    cbind(convert_and_get_ci(estim_advi, z = qnorm(0.5 + 0.5 * ci_level)), "algorithm" = "Full-rank VI")
    )

  #####
  # Trace of covariance as measure of overall scale
  trace_of_cov <- mean(sapply(seq_along(laplace_out$ys), function(i) sum(diag(cov(laplace_out$ys[[i]])))))
  plt_df <- cbind(plt_df, "trace_of_cov" = trace_of_cov)
  
  # Save output
  save(plt_df, var_improvement, file = paste0("bias_logistic_", parameters$dataset,".RData"))
  save(R, sgld_out, sgldcv_out, laplace_out, advi_out, file = paste0("bias_logistic_", parameters$dataset,"_output.RData"))
  return(plt_df)
}

## Run simulations and save output
pima_df <- run_logistic_experiment(pima_parameters)
ds1_df <- run_logistic_experiment(ds1_parameters)

plt_df <- rbind(cbind(pima_df, dataset = "(a) Pima dataset"),
                cbind(ds1_df, dataset = "(b) DS1 dataset"))
save(plt_df, file = "bias_logistic.RData")

## Plot data
load(file = "bias_logistic.RData")
plt_df$algorithm <- factor(plt_df$algorithm, level = c("SGLD", "SGLD-cv", "Laplace", "Full-rank VI"))
plt_df$Estimator <- factor(plt_df$Estimator, level = c("V", "L", "Coupling", "Tractable lower bound"))

levels(plt_df$Estimator)[3] <- "Coupling bound"

library(ggplot2)

plt <- 
  ggplot(plt_df, aes(y = w2sq, 
                   x = algorithm, 
                   color = Estimator)) +
  facet_wrap(~dataset, scales = "free") +
  geom_point() +
  scale_y_log10(labels = function(x) format(x, scientific = TRUE)) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_up), width = 0.3) +
  labs(x = NULL, y = "Squared Wasserstein distance") +
  geom_hline(aes(yintercept = trace_of_cov, linetype = "dashed")) +
  theme_bw(base_size = 12) +
  scale_color_discrete(name = NULL,
                       breaks = c("V", "L", "Coupling bound", "Tractable lower bound")) +
  scale_linetype_manual(name = NULL, 
                        values = c("dashed" = "dashed"), 
                        labels = c("Trace of posterior covariance")) +
  theme(strip.text.x = element_text(hjust = 0, margin=margin(l=0,b=4), size = 12),
        strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  guides(colour = guide_legend(order = 1), 
         linetype = guide_legend(order = 2))
plt

ggsave(filename = "bias_logistic.pdf",
       plot = plt,
       width = 24, height = 10, units = "cm")
