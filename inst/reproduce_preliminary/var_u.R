#####
# Benchmark variance estimation methods for the proposed debiased estimators.
#####

library(wassersteinbound)
library(doParallel)
library(rngtools)
library(reshape2)
library(dplyr)
library(ggplot2)
library(ggh4x)
library(latex2exp)
source("helpers.R")

# Experiment parameters ####
jack_reps <- 100
true_reps <- 5000
D <- 10^c(1,2)
N <- 10^c(1,2,3)
Sigma_sq <- c(1.1, 2, 10)

seed <- 12345
set.seed(seed)
rng <- RNGseq(max(jack_reps, true_reps), seed)

ncores <- 4
#####

# Variance estimators ####
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)

debiased_out <-
  foreach(d = D, .combine = "rbind") %:%
  foreach(n = N, .combine = "rbind") %:%
  foreach(sigma_sq = Sigma_sq, .combine = "rbind") %:%
  foreach(i = 1:jack_reps, rng_ = rng, .combine = "rbind", .packages = c("wassersteinbound")) %dopar% {
    # Set RNG
    rngtools::setRNG(rng_)
    
    xprime <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = n, ncol = d)
    yprime <- matrix(rnorm(n * d), nrow = n, ncol = d)
    y <- matrix(rnorm(n * d), nrow = n, ncol = d)
    
    w2sq_xprimey <- w2sq_empirical_jack(xprime,y)
    w2sq_yprimey <- w2sq_empirical_jack(yprime,y)
    
    return(data.frame("d" = d, "n" = n, "sigma_sq" = sigma_sq, 
                      "Jackknife" = var(w2sq_xprimey$jackknife_assignment_costs - w2sq_yprimey$jackknife_assignment_costs) / n, 
                      "Consistent" = var(w2sq_xprimey$potentials_x + w2sq_xprimey$potentials_y - w2sq_yprimey$potentials_x - w2sq_yprimey$potentials_y) / n))
  }
parallel::stopCluster(cl)

debiased_var <- melt(debiased_out, 
                     id.vars = c("d", "n", "sigma_sq"), 
                     measure.vars = c("Jackknife", "Consistent"),
                     value.name = "Variance", variable.name = "Estimator")
debiased_var$Estimator <- factor(debiased_var$Estimator,levels=c("Jackknife","Consistent"))
#####

# Estimate true variance ####
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)

true_out <-
  foreach(d = D, .combine = "rbind") %:%
  foreach(n = N, .combine = "rbind") %:%
  foreach(sigma_sq = Sigma_sq, .combine = "rbind") %:%
  foreach(i = 1:true_reps, rng_ = rng, .combine = "rbind", .packages = c("wassersteinbound")) %dopar% {
    # Set RNG
    rngtools::setRNG(rng_)
    
    xprime <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = n, ncol = d)
    yprime <- matrix(rnorm(n * d), nrow = n, ncol = d)
    y <- matrix(rnorm(n * d), nrow = n, ncol = d)
    
    w2sq_xprimey <- w2sq_empirical(xprime,y)
    w2sq_yprimey <- w2sq_empirical(yprime,y)
    
    return(data.frame("d" = d, "n" = n, "sigma_sq" = sigma_sq, 
                      "Unbiased" = w2sq_xprimey$w2sq - w2sq_yprimey$w2sq))
  }
parallel::stopCluster(cl)

true_var <- get_true_var(true_out)
true_var_boot_ci <- get_true_var_boot_ci(true_out, true_reps)

true_var_df <- true_var
true_var_df$ci_up <- true_var_boot_ci$ci_up
true_var_df$ci_lo <- true_var_boot_ci$ci_lo
#####
save(debiased_out, debiased_var, true_out, true_var_df, true_reps, file = "var_u.RData")

# Plot and save ####
load("var_u.RData")

# plt_var <-  
# get_box_plots(debiased_var, true_var_df) + 
#   scale_fill_discrete(breaks = c("Jackknife", "Consistent", "Naive"))
# plt_var

library(scales)

plt_var <- 
  ggplot(debiased_var) +
  facet_nested(d~sigma_sq+n, scales = "free", independent = "y",
               labeller = labeller(sigma_sq = sigma_sq_label, d = d_label, n = n_label)) +
  geom_hline(true_var_df, mapping = aes(yintercept = Variance, linetype = "Ground truth")) +
  geom_rect(true_var_df, mapping = aes(xmin = -Inf, xmax = Inf, ymin=ci_lo, ymax=ci_up), alpha = 0.2) +
  geom_boxplot(mapping = aes(x = Estimator, y = Variance, color = Estimator, group = interaction(n, Estimator))) +
  theme_bw() +
  labs(x = NULL, linetype = NULL) +
  scale_y_continuous(minor_breaks = NULL, breaks = pretty_breaks(n=5)) +
  scale_x_discrete(breaks=NULL) +
  guides(color = guide_legend(order = 1), linetype = guide_legend(order = 2)) +
  theme(panel.grid = element_blank())
plt_var


ggsave(filename = "var_u.pdf",
       plot = plt_var,
       width = 28, height = 8, units = "cm")
# #####





