library(wassersteinbound)
library(ars)
library(reshape2)
library(ggplot2)
library(ggh4x)
library(ggridges)

# Draw n samples from d-dimensional multivariate logistic distribution,
# density pi(x) proportional to logistic(||x||) = exp(-||x||)/ (1 + exp(-||x||))^2
DrawFromLogistic <- function(n, d) {
  # 1. Sample radial component
  if(d == 1) {
    radials <- rlogis(n)
  } else { # Use adaptive rejection sampling
    # Radial density from Definition 1 & Theorem 3 in:
    # Gomez et al (2003). "A survey on continuous elliptical vector distributions"
    # https://www.mat.ucm.es/serv/revmat/vol16-1/vol16-1k.pdf
    log_radial <- function(r) return((d - 1) * log(r) - r - 2 * log1p(exp(-r)))
    log_radial_deriv <- function(r) return((d - 1) / r + 1 - 2 / (1 + exp(-r)))
    
    radials <- ars::ars(n, f = log_radial, fprima = log_radial_deriv, x = c(0.1, 1, d - 1, 11, 2*d), m = 5, lb = TRUE, xlb = 0)
  }
  
  # 2. Sample angular component
  normalize <- function(x) return(x / sqrt(sum(x^2)))
  
  draws <- vector("list", n)
  for(i in 1:n) draws[[i]] <- radials[i] * normalize(rnorm(d))
  
  return(draws)
}

# Randomness
seed <- 12345
SetSeed_pcg32(seed)
set.seed(seed)

# Run the whole experiment
run_experiment <- function(d, h, R, probs, iter, thin = 1, acc_rate_iters = 1e4) {
  target <- list("target_type" = "multivariate_logistic")
  
  sampler_params <- list("Sigma" = rep(h^2,d))
  iters <- seq(0, iter, thin)
  
  # Simulate chain and store ||x|| at each iteration
  sample_norm <- function(x0) {
    mcmc_out <- rwm_cpp(target, sampler_params, x0, iter, thin)
    norms <- sqrt(rowSums(mcmc_out$xs^2))
    return(norms)
  }
  
  # Start from overdispersed version of the target
  x0s <- DrawFromLogistic(R, d) 
  x0s <- lapply(x0s, function(x) 2 * x)
  
  # Simulate R chains
  mcmc_out_norms <- do.call(rbind, lapply(x0s, sample_norm))
  
  # # Compute iteration-wise quantiles for ||x||
  # quantiles <- apply(mcmc_out_norms, 2, function(x){quantile(x, probs = probs)}) # Each probability populates a row
  # rownames(quantiles) <- probs 
  # colnames(quantiles) <- iters # Each iteration populates a column
  # quantiles <- reshape2::melt(quantiles, varnames = c("Probability", "Iteration"), value.name = "Quantile")
  
  # Calculate the acceptance rate
  acc_rate <- rwm_cpp(target, sampler_params, x0s[[1]], acc_rate_iters, acc_rate_iters)$acc_rate_x
  
  return(list("acc_rate" = acc_rate,
              "mcmc_out" = mcmc_out_norms,
              "iters" = iters,
              "d" = d))
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

# d = 1 ####
d    <- 1    # Dimension
h    <- 3    # Step size
R    <- 1e5  # Number of chains
iter <- 14   # Iteration count
thin <- 2

d1_out <- run_experiment(d, h, R, probs, iter, thin)
d1_out$acc_rate
#####

# d = 10 ####
d    <- 10   # Dimension
h    <- 2.5  # Step size
R    <- 1e5  # Number of chains
iter <- 140  # Iteration count
thin <- 20

d10_out <- run_experiment(d, h, R, probs, iter, thin)
d10_out$acc_rate
#####


# Plot ####
d1_df <- melt_df(d1_out)
d10_df <- melt_df(d10_out)

# Manually truncate to zoom in on y-axes
plt_df <- rbind(cbind(d1_df[d1_df$State < 15,], "d" = "(a) Dimension d = 1"),
                cbind(d10_df[d10_df$State < 45,], "d" = "(a) Dimension d = 10"))


plt <-
  ggplot(data = plt_df, aes(x = State, y = factor(Iteration), 
                            fill = factor(stat(quantile))))+
  ggridges::stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE,
                                quantiles = 5, quantile_lines = TRUE, scale = 1.1,
                                rel_min_height = 0.005) +
  ggh4x::facet_grid2(~d, scales="free", independent = "y") +
  scale_fill_viridis_d(name = "Quintile") +
  coord_flip() +
  guides(fill = guide_legend(override.aes = list(linetype = 0))) +
  theme_bw() +
  labs(y = "Iteration", x = "Radius") +
  theme(strip.text.x = element_text(hjust = 0, margin=margin(l=0,b=4), size = 11),
        strip.background = element_blank(),
        panel.grid.minor = element_blank())
plt

ggsave(filename = "rwm_logistic.pdf",
       plot = plt,
       width = 24, height = 6, units = "cm")
#####