library(wassersteinbound)
library(doParallel)
library(Matrix)
library(ggplot2)

# Helpers #####
source("../helpers.R")

# Covariance matrix of a periodic-boundary AR(1) process with mean zero,
# autocorrelation "alpha" and increment standard deviation "sigma".
generate_periodic_ar1_covariance <- function(d, alpha, sigma) {
  Sigma <- matrix(NA, ncol = d, nrow = d)
  
  if(d == 1) {
    Sigma[1, 1] <- sigma^2 / (1 - alpha^2)
  } else {
    entry <- function(i,j){ k <- abs(i - j); return(alpha^k + alpha^(d - k))}
    Sigma <- outer(1:d, 1:d, FUN = entry)
    
    rescale <- sigma^2 / ((1 - alpha^d) * (1 - alpha^2))
    Sigma <- rescale * Sigma
  }
  
  return(Sigma)
}

# Deterministic update Gibbs sampler writes as a vector autoregressive process 
# with slope matrix B.
#
# See Roberts and Sahu (1997), Section 2.2
gibbs_slope_matrix <- function(Sigma){
  d <- nrow(Sigma)
  
  # Eqn. (3) of Roberts and Sahu (1997)
  Q <- solve(Sigma)
  A <- diag(d) - diag(1/diag(Q)) %*% Q # The diagonal of this matrix is NULL.
  
  I_minus_L <- diag(d) - Matrix::tril(A) # Doesn't matter if we include the diagonal or not.
  U <- Matrix::triu(A)
  
  # Eqn. (4) of Roberts and Sahu (1997)
  B <- forwardsolve(I_minus_L, U)
  return(B)
}

#########
# Setup #
#########


# Target parameters #####
d     <- 50L   # Dimension
sig   <- 1     # Noise std. deviation (no effect on the algorithm)
alpha <- 0.95  # Correlation

# Target mean and covariance
mu <- rep(0, d)
Sigma <- generate_periodic_ar1_covariance(d, alpha, sig)

# Gibbs sampler parameters
std_dev <- sqrt(sig^2 / (1 + alpha^2))
c <- alpha / (1 + alpha^2)
#####

run_gibbs_experiment <- function(mu_0 = mu, Sigma_0,
                                 mu_ = mu, Sigma_ = Sigma,  # Target parameters
                                 iter = 5e3, thin = 5, # Iterations and Thinning factor
                                 R = 1024,   # Number of chains
                                 std_dev_ = std_dev, c_ = c, # Parameters for MCMC sampler
                                 debias = c(2000, 5000), conf_level = 0.95, boot_reps = 1e3, # Parameters for estimators
                                 seed = 12345, ncores = 4){
  
  # RNG and parallel processing
  set.seed(seed); SetSeed_pcg32(seed)
  
 
  # Sample ####
  iters <- seq(0,iter,thin)
  
  # Initialization
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  L <- t(chol(Sigma_))
  L_0 <- t(chol(Sigma_0))
  
  x_infty <- y0 <- vector(mode = "list", R)
  for(r in 1:R) {
    x_infty[[r]] <- as.vector(L %*% rnorm(d))
    y0[[r]] <- as.vector(L_0 %*% rnorm(d))
  }

  # Coupled MCMC
  crn_out <- foreach(i = 1:R, x0_ = x_infty, y0_ = y0, .packages = "wassersteinbound") %do% {
    SetSeed_pcg32(seed, i)
    gibbs_periodicAR1_CRN_cpp(x0_, y0_, std_dev_, c_, iter, thin)$squaredist
  }
  reflmax_out <- foreach(i = 1:R, x0_ = x_infty, y0_ = y0, .packages = "wassersteinbound") %do% {
    SetSeed_pcg32(seed, i)
    gibbs_periodicAR1_ReflMax_cpp(x0_, y0_, std_dev_, c_, iter, thin)$squaredist
  }

  # Single-chain MCMC
  singlechain_out <- foreach(i = 1:R, y0_ = y0, 
                             .combine = rbind_within_list, .multicombine = T,
                             .init = vector("list", length(iters)), 
                             .packages = c("wassersteinbound")) %dopar% {
                               SetSeed_pcg32(seed, i)
                               asplit(gibbs_periodicAR1_cpp(y0_, std_dev_, c_, iter, thin)$xs, 1)
                             }
  parallel::stopCluster(cl)
  
  # Get bounds #####
  # Empirical bound
  w2sq_empirical_estimates <- get_empirical_w2sq(singlechain_out, do.call("rbind", x_infty), ncores)
  empirical_df <- get_empirical_bounds(w2sq_empirical_estimates, iters, debias, conf_level)
  
  # Exact squared Wasserstein distance
  B <- gibbs_slope_matrix(Sigma_) # "Slope" matrix
  w2sq_exact <- w2sq_convergence_gaussian_recursive_cpp(mu_0, Sigma_0, mu_, Sigma_, B, ncol(B), iter, thin)$w2sq
  exact_df <- data.frame("iter"  = iters, "w2sq"  = w2sq_exact, "ci_up" = NA, "ci_lo" = NA, "estimator" = "Exact squared Wasserstein")

  # Coupling bound
  crn_df <- get_coupling_bound(crn_out, iters, name = "Coupling bound (CRN)", conf_level, boot_reps)
  reflmax_df <- get_coupling_bound(reflmax_out, iters, name = "Coupling bound (ReflMax)", conf_level, boot_reps)

  return(rbind(empirical_df, crn_df, reflmax_df, exact_df))
}

plt_df <- rbind(
  cbind(run_gibbs_experiment(Sigma_0 = 4 * Sigma), "init" = "(a) Overdispersed start"), 
  cbind(run_gibbs_experiment(Sigma_0 = diag(diag(Sigma))), "init" = "(b) Naive start")
  )


log_const_p <- log.const.p(20)


### Manual plot fiddling
levels <- c("U", "L", "Coupling bound (CRN)", "Coupling bound (ReflMax)", "Exact squared Wasserstein")


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
colors <- c(gg_color_hue(4), "black")
fills <- c(gg_color_hue(4), NA)
###


plt <-
  ggplot(plt_df, aes(x = iter, y = w2sq, color = estimator, fill = estimator)) +
  facet_grid(~init) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_line() +
  geom_ribbon(aes(ymin = ci_up, ymax = ci_lo),
                    alpha = 0.15, colour = NA) +
  labs(x = "Iteration", y = "Squared Wasserstein distance", color = "Estimator") +
  scale_x_continuous(limits = c(0, 2200)) +
  scale_y_continuous(trans = log_const_p,
                     breaks = c(0,10,30,100,300,1000,3000), labels = scales::label_number()) +
  coord_cartesian(ylim = c(-5, 1000)) +
  scale_color_manual(name = NULL, values = colors, breaks = levels) +
  scale_fill_manual(name = NULL, values = fills, na.value="transparent", breaks = levels) +
  theme_bw() +
  theme(panel.grid.minor = element_blank(),
        #panel.spacing = unit(0.75, "lines"),
        strip.text.x = element_text(hjust = 0, margin=margin(l=0,b=4), size = 11),
        strip.background = element_blank())
plt


# Save plot
ggsave(filename = "gibbs.pdf",
       plot = plt,
       width = 24, height = 8, units = "cm")



