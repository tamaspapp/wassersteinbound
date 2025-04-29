library(wassersteinbound)
library(reshape2)
library(dplyr)
library(ggplot2)
library(ggridges)
library(ggpubr)
library(latex2exp)

DrawFromGaussianMixture <- function(n, p, mu, sig) {
  k <- sample.int(length(p), n, replace = TRUE, prob = p) # Draw mixture components
  xs <- rnorm(n, mu[k], sig[k]) # Draw samples conditional on the mixture components
  return(xs)
}

GaussianMixtureDensity <- function(x, ps, mus, sigmas) {
  ps <- ps / sum(ps) # Normalize the probabilities just in case
  mixture_dens <- function(x, ps, mus, sigmas) return(sum(ps * dnorm(x, mean = mus, sd = sigmas)))
  mixture_dens_vec <- Vectorize(function(x) mixture_dens(x, ps, mus, sigmas), "x")
  
  return(mixture_dens_vec(x))
}

# Run MCMC
run_experiment <- function(p, mu, sig, # 1d mixture target parameters
                           x0s, # list of starting values
                           h, iter, thin, # MCMC parameters
                           acc_rate_iters = 1e4) {
  
  target <- list("p" = p, "mu" = mu, "sigma" = sig, "target_type" = "1d_gaussian_mixture")
  
  sampler_params <- list("Sigma" = rep(h^2,1))
  iters <- seq(0, iter, thin)
  
  # Simulate chain and store one-dimensional x at each iteration
  sample <- function(x0) {
    mcmc_out <- rwm_cpp(target, sampler_params, x0, iter, thin)
    return(as.vector(mcmc_out$xs))
  }
  
  # Simulate R = length(x0s) chains
  mcmc_out <- do.call(rbind, lapply(x0s, sample))
  
  # Calculate the acceptance rate
  acc_rate <- rwm_cpp(target, sampler_params, x0s[[1]], acc_rate_iters, acc_rate_iters)$acc_rate_x
  
  return(list("acc_rate" = acc_rate,
              "mcmc_out" = mcmc_out,
              "iters" = iters,
              "p" = p,
              "mu" = mu,
              "sig" = sig))
}

melt_df <- function(mcmc_out){
  iters <- mcmc_out$iters
  R <- nrow(mcmc_out$mcmc_out)
  
  datafr <- mcmc_out$mcmc_out
  colnames(datafr) <- mcmc_out$iters # Iteration
  rownames(datafr) <- seq(1,R) # Replicate
  datafr <- reshape2::melt(datafr, varnames = c("Replicate", "Iteration"), value.name = c("State"))
  
  return(datafr)
}

# Get Wasserstein distance and empirical estimator at sample size 1
get_wasserstein <- function(mcmc_out) {
  iters <- mcmc_out$iters
  R <- nrow(mcmc_out$mcmc_out)
  p <- mcmc_out$p
  mu <- mcmc_out$mu
  sig <- mcmc_out$sig
  
  pi_samples <- DrawFromGaussianMixture(R, p, mu, sig)
  pi_t_samples <- mcmc_out$mcmc_out
  
  # Empirical bound, at sample size n=1 ####
  pi_t_means <- colMeans(pi_t_samples)
  pi_mean <- mean(pi_samples)
  
  pi_t_sqnorms <- colMeans(pi_t_samples^2) 
  pi_sqnorm <- mean(pi_samples^2)
  
  pi_t_vars <- apply(pi_t_samples,2,var)
  pi_var <- var(pi_samples)
  
  # w2sq_empirical_n1 <- (pi_t_means - pi_mean)^2 + abs(pi_t_sqnorms - pi_sqnorm)
  plugin <- pi_t_sqnorms + pi_sqnorm - 2 * pi_t_means * pi_mean
  debias <- 2 * pmin(pi_t_vars, pi_var)
  
  w2sq_empirical_n1 <- plugin - debias
  #####
  
  # True Wasserstein distance, with some small positive bias ####
  pi_t_sorted <- apply(pi_t_samples, 2, sort)
  pi_sorted <- sort(pi_samples)
  w2sq_true <- colMeans(sweep(pi_t_sorted, 1, pi_sorted)^2)
  #####
  
  df <- data.frame("Exact" = sqrt(w2sq_true),
                   "Empirical" = sqrt(w2sq_empirical_n1), 
                   "Iteration" = iters)
  df <- reshape2::melt(df, value.name = "Distance", id.vars = "Iteration", variable.name = "Estimator")
  
  return(df)
}

# Randomness
seed <- 12345
SetSeed_pcg32(seed)
set.seed(seed)

##########################
# Well-separated mixture #
##########################
p  <- c(0.5, 0.5) # Mixture proportions
mu <- c(-5, 5)    # Mixture means
sig <- c(1, 1)    # Mixture standard deviations

# Plot the target density
x_pdf <- seq(-10, 10, by = 0.05)
y_pdf <- GaussianMixtureDensity(x_pdf, p, mu, sig)
qplot(x_pdf, y_pdf,geom = "line", xlab = "x", ylab = "Density", main = "Multimodal target") + theme_bw()

# Diffusive MCMC: small step size #####
iter <- 10    # Iteration count 
thin <- 2
h    <- 2     # Step size
R    <- 2e5   # Replicate count

# Start from outwardly shifted version of the target
x0s <- as.list(DrawFromGaussianMixture(R, p, 2 * mu, sig))

out1 <- run_experiment(p, mu, sig, x0s, h, iter, thin)
out1$acc_rate
#####

# Less diffusive MCMC: larger step size #####
iter <- 10    # Iteration count
thin <- 2
h    <- 6     # Step size
R    <- 2e5   # Replicate count

# Start from outwardly shifted version of the target
x0s <- as.list(DrawFromGaussianMixture(R, p, 2 * mu, sig))

out2 <- run_experiment(p, mu, sig, x0s, h, iter, thin)
out2$acc_rate
#####

# Step size in between, plus start in one of the modes #####
iter <- 100   # Iteration count 
thin <- 20
h    <- 4     # Step size
R    <- 4e5   # Replicate count

# Start in the right-most mode
x0s <- as.list(rnorm(R, 5, 2))

out3 <- run_experiment(p, mu, sig, x0s, h, iter, thin)
out3$acc_rate
#####

# Plot ####
df1 <- melt_df(out1)
df2 <- melt_df(out2)
df3 <- melt_df(out3)

density_df <- rbind(
  cbind(df1, "setting" = "(a) Overdispersed, diffusive"),
  cbind(df2, "setting" = "(b) Overdispersed, not diffusive"),
  cbind(df3, "setting" = "(c) Not overdispersed, diffusive"))
density_df$y_type = "State"

w1 <- get_wasserstein(out1)
w2 <- get_wasserstein(out2)
w3 <- get_wasserstein(out3)

wasserstein_df <- rbind(
  cbind(w2, "setting" = "(a) Overdispersed, diffusive"),
  cbind(w2, "setting" = "(b) Overdispersed, not diffusive"),
  cbind(w3, "setting" = "(c) Not overdispersed, diffusive"))
wasserstein_df$y_type = "Wasserstein distance"

plt_df <- dplyr::bind_rows(density_df, wasserstein_df)

plt <-
  ggplot(plt_df) +
  coord_flip() +
  facet_grid(y_type ~ setting, scales = "free", switch = "y") + 
  labs(y = "Iteration", x = "", colour = NULL) +
  geom_vline(data = subset(plt_df,y_type == "Wasserstein distance"), 
             aes(xintercept = 0), lty = 2) +
  geom_line(data = subset(plt_df,y_type == "Wasserstein distance"),
            aes(x = Distance, y = factor(Iteration), colour = Estimator, group = Estimator)) +
  geom_point(data = subset(plt_df,y_type == "Wasserstein distance"),
            aes(x = Distance, y = factor(Iteration), colour = Estimator, group = Estimator)) +
  ggridges::stat_density_ridges(data = subset(plt_df,y_type == "State"),
    aes(x = State, y = factor(Iteration), fill = factor(stat(quantile))),
    geom = "density_ridges_gradient", calc_ecdf = TRUE,
    quantiles = 5, quantile_lines = TRUE, scale = 1.1, bandwidth = 0.2, rel_min_height = 0.005) +
  scale_fill_viridis_d(name = "Quintile") +
  # scale_color_discrete(labels = c("Exact Wasserstein", "E[V] at n=1")) +
  scale_color_manual(labels = c("Exact Wasserstein", TeX("${E[V]}^{1/2}$ at n=1        ")), 
                     values = c("black", "red")) +
  theme_bw() +
  theme(strip.placement = "outside",
        strip.text.y = element_text(size = 11),
        strip.text.x = element_text(size = 11, hjust = 0, margin=margin(l=0,b=4)),
        strip.background = element_blank(),
        panel.grid.minor = element_blank()) +
  guides(fill = guide_legend(override.aes = list(linetype = 0), order = 1),
         colour = guide_legend(order = 2)) +
  scale_y_discrete(expand = c(0, 0.2))
plt

ggsave(filename = "rwm_mixture.pdf",
       plot = plt,
       width = 24, height = 10, units = "cm")
