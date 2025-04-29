
library(wassersteinbound)
library(Matrix)
library(doParallel)
library(rngtools)
library(reshape2)
library(dplyr)
library(ggplot2)

###
# 0. Setup
###

# Target parameters: autocorrelation and dimension
rho <- 0.5
ds <- rev(c(128, 256, 512, 1024))#rev(c(100, 400, 700, 1000))

# Step sizes
hs <- ds^(-1/6)

# Collate parameters associated to each dimension
problem_params <- foreach(d = ds, h = hs) %do% {
    # Get exact target parameters
    mu <- rep(0, d)
    target_params <- AR1_target_params(d, rho)
    
    Sigma        <- target_params$Sigma
    Sigma_chol_u <- chol(Sigma)
    Omega        <- target_params$Omega
    Omega_chol_u <- target_params$Omega_chol_u
    
    # Get ULA target parameters
    Sigma_ULA <- as(solve((diag(d) - h^2 / 4 * Omega) %*% Omega), "matrix")
    Sigma_ULA_chol_u <- chol(Sigma_ULA)
    
    # Get friction parameters
    gammas <- c("ULA" = Inf, 
                "OBABO" = 2 / max(eigen(Sigma,T,T)$values)) # Critical damping
    
    list("target_type"  = "sparse_gaussian",
         "mu"           = mu,
         "Sigma"        = Sigma,
         "Omega"        = Omega,
         "Omega_chol_u" = Omega_chol_u,
         "Sigma_chol_u" = Sigma_chol_u,
         "h"                = h,
         "d"                = d,
         "gammas"           = gammas,
         "Sigma_ULA"        = Sigma_ULA, 
         "Sigma_ULA_chol_u" = Sigma_ULA_chol_u,
         "eigenSigma" = eigen(Sigma,T,T)$values,
         "eigenSigma_ULA"= eigen(Sigma_ULA,T,T)$values)
  }

# Sample sizes and replicate count
n <- 1024 # Samples used in each empirical estimator
reps <- 256 # Repeats of the empirical estimator, to get variance estimates
n_coupled_chains <- 10
# Set up RNG
seed <- 12345
set.seed(seed)
rng <- RNGseq(max(reps, n_coupled_chains), seed)

# Parallel computing
ncores <- 8
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)


###
# 1. Exact Wasserstein distance
###
w2sq_exact <- 
  foreach(params_ = problem_params, .combine = "rbind", .packages = c("wassersteinbound", "Matrix")) %dopar%
    {
      d <- params_$d
      mu <- params_$mu
      Sigma <- params_$Sigma
      Sigma_ULA <- params_$Sigma_ULA
      
      data.frame("d" = d, "w2sq" = w2sq_gaussian_cpp(mu, Sigma_ULA, mu, Sigma), "estimator" = "Exact squared Wasserstein")
    }

###
# 2. Empirical estimators
###
signed_square <- function(x) { return(sign(x) * x^2)}

correlation <- c(0, 0.4, 0.8)

w2sq_empirical <- 
  foreach(corr = correlation, .combine = "rbind") %:%
  foreach(params = problem_params, .combine = "rbind") %:%
  foreach(i = 1:reps, rng_ = rng, .combine = "rbind", .packages = c("wassersteinbound", "Matrix")) %dopar% {
        # Set RNG
        rngtools::setRNG(rng_)
        
        # Sample empirical measures. We know their stationary distributions, so skip MCMC altogether.
      d <- params$d
      eigenSigma_ULA <- params$eigenSigma_ULA
      eigenSigma <- params$eigenSigma
      
      z1ula <- matrix(rnorm(n * d), nrow = n, byrow = T)
      z1mala <- corr * z1ula + sqrt(1-corr^2) * matrix(rnorm(n * d), nrow = n, byrow = T)
      
      ula1 <- sweep(z1ula,2,sqrt(eigenSigma_ULA),"*") # Effect: multiply each row of z1 by vector sqrt(diagSigmaULA)
      mala1 <- sweep(z1mala,2,sqrt(eigenSigma),"*")
      mala2 <- matrix(rnorm(n * d, sd = sqrt(eigenSigma)), nrow = n, byrow = T)
      
        # Calculate empirical estimators
        timing <- Sys.time()
        
        plugin <- w2sq_empirical(mala2, ula1)$w2sq
        debias <- w2sq_empirical(mala2, mala1)$w2sq

        timing <- Sys.time() - timing
        
        # Calculate proposed estimators
        u <- plugin - debias
        l <- signed_square(sqrt(plugin) - sqrt(debias))
        
        # Output alongside dimension
        data.frame("d" = d, "U" = u, "L" = l, "walltime" = timing, "correlation" = corr)
      } 

# Melt data to long format
w2sq_empirical <- melt(w2sq_empirical, id.vars = c("d", "correlation"), measure.vars = c("U", "L"),
                       value.name = "w2sq", variable.name = "estimator")

w2sq_empirical_std_dev <-
  w2sq_empirical %>%
  dplyr::group_by(estimator, d, correlation) %>%
  dplyr::summarize(w2sq.sd = sd(w2sq))

variance <-
  # Summarize: get groupwise variance
  w2sq_empirical %>%
  dplyr::group_by(estimator, d, correlation) %>%
  dplyr::summarize(var = var(w2sq)) %>%
  # Add column with groupwise factor of reduction and improvement
  dplyr::group_by(estimator, d) %>%
  dplyr::mutate(var_reduction_factor = var/var[correlation == 0]) %>%
  dplyr::group_by(estimator, d) %>%
  dplyr::mutate(var_improvement_factor = 1/var_reduction_factor)
variance$d <- factor(variance$d)

print("Variance reduction factors:")
variance


###
# 3. Coupling-based estimators
###
iter <- 3e3
burn_in <- 1e3

w2sq_coupling <- 
  foreach(params_ = problem_params, .combine = "rbind") %:%
  foreach(gamma_ = params_$gammas, gamma_label = names(params_$gammas), .combine = "rbind") %:%
  foreach(i = 1:n_coupled_chains, rng_ = rng, .combine = "rbind", .packages = c("wassersteinbound", "Matrix")) %dopar% {
        # Set RNG
        rngtools::setRNG(rng_)
        SetSeed_pcg32(seed, i)
        
        # Start the coupled chains independently from their respective stationary distributions.
        d <- params_$d
        x0_ <- as.vector(rnorm(d) %*% params_$Sigma_ULA_chol_u)
        y0_ <- as.vector(rnorm(d) %*% params_$Sigma_chol_u)
        
        # Sample
        sampler_params <- list("gamma" = gamma_, "delta" = params_$h, "Sigma" = rep(1,d))
        
        timing <- Sys.time()
        cpl_out <- obab_horowitz_CRN_cpp(params_, sampler_params, x0_, y0_, iter, thin = 1)
        timing <- Sys.time() - timing
        
        data.frame("d" = d, 
                   "w2sq" = mean(cpl_out$squaredist[(burn_in + 1):iter]), 
                   "walltime" = timing, 
                   "acc_rate" = mean(cpl_out$acceptances_y),
                   "estimator" = paste0("Coupling (", gamma_label,")"))
      }
# Get computing time and acceptance rates for coupling bounds
coupling_acc_rate <- w2sq_coupling$acc_rate
coupling_time <- w2sq_coupling$walltime
# Convert data to right format
w2sq_coupling <- select(w2sq_coupling, c("d", "w2sq", "estimator"))

save(w2sq_empirical, w2sq_coupling, w2sq_exact,
     reps, n, n_coupled_chains, problem_params, ds,
     file = "bias_ar1.RData")

parallel::stopCluster(cl)

###
# 4. Plot
###
load(file = "bias_ar1.RData")

w2sq_empirical <- w2sq_empirical[w2sq_empirical$correlation==0,]
w2sq_empirical$correlation <- NULL

df_plt <- rbind(w2sq_empirical, w2sq_coupling, w2sq_exact)

# Get mean, standard error, standard deviation
df_plt <- df_plt %>%
  dplyr::group_by(estimator, d) %>%
  dplyr::summarize(w2sq.mean = mean(w2sq), w2sq.se = sd(w2sq) / sqrt(reps), w2sq.sd = sd(w2sq))

# Add trace of covariance
trace_Sigma <- data.frame("estimator" = "Trace of target covariance",
                          "d" = ds,
                          "w2sq.mean" = sapply(problem_params, function(x) sum(diag(x$Sigma))),
                          "w2sq.se" = NA, "w2sq.sd" = NA)
df_plt <- rbind(df_plt, trace_Sigma)

levels <- c("U", "L", "Coupling (ULA)", "Coupling (OBABO)", "Exact squared Wasserstein", "Trace of target covariance")
df_plt$estimator <- factor(df_plt$estimator,levels=levels)

# Changing notation in plot
levels[c(3,4)] <- c("Coupling bound (ULA)", "Coupling bound (OBABO)")
levels(df_plt$estimator) <- levels

# Manual legend fiddling ####
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

linetypes <- c(rep("solid", 5),"dashed")
colors <- c(gg_color_hue(4), "black", "black")
fills <- c(gg_color_hue(4), NA, NA)
names(linetypes) <- names(colors) <- names(fills) <- levels
#####

plt <- 
  ggplot(df_plt, aes(x = d, color = estimator, linetype=estimator, fill = estimator)) +
  # Mean and two standard deviations
  geom_line(aes(y = w2sq.mean)) +
  geom_ribbon(aes(ymin = w2sq.mean - 2*w2sq.sd, ymax =  w2sq.mean + 2*w2sq.sd), alpha = 0.1, color = NA) +
  # Axis scales
  scale_y_log10(breaks = scales::breaks_log(7), minor_breaks = NULL) +
  scale_x_log10(breaks = ds, minor_breaks = NULL) +
  #scale_x_continuous(breaks = ds) +
  # Legend
  scale_color_manual(values = colors, name=NULL) +
  scale_fill_manual(values = fills, na.value="transparent",name=NULL) +
  scale_linetype_manual(values = linetypes, name=NULL) +
  # Axis and legend labels
  labs(x = "Dimension", y = "Squared Wasserstein distance") +
  # Plot theme
  theme_bw() +
  theme(panel.grid.minor = element_blank())
plt


ggsave(filename = "bias_langevin.pdf",
       plot = plt,
       width = 18, height = 9, units = "cm")
