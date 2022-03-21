library(wassersteinbound)
library(doParallel)

##################
# Setup ##########
##################

#####
# Set up RNGs and number of clusters
seed <- 12345L
set.seed(seed)
SetSeed_cpp(seed)

ncores <- parallel::detectCores()

#####
# Target parameters
d     <- 50L   # Dimension
sig   <- 1     # Noise std. deviation (no effect on algorithm)
alpha <- 0.95  # Correlation

Sigma <- GenerateTargetCovarianceMatrix_ar1(d, alpha, sig)

std_dev <- sqrt(sig^2 / (1 + alpha^2))
c <- alpha / (1 + alpha^2)

#####
# MCMC parameters
S <- 4 * Sigma        # starting covariance
t_chol_S <- t(chol(S))

iter_store_x <- 5000L
iter_final_x <- 5000L

L <- iter_final_x

thin  <- 5L

start_debias <- 2000L
end_debias   <- 4000L

debias_iters <- seq(floor(start_debias / thin) + 1, floor(end_debias / thin) + 1)
which_iters <- seq(0, iter_store_x, thin)


#####
#  Replicate count
R <- 1000L

###########################
# Get exact W2^2 ##########
###########################

mu_0 <- mu <- rep(0, d) 

iter_final_exactw2sq <- 3000L
thin_exactw2sq       <- 1L

w2sq_exact <- EvaluateW2sqGibbs(mu_0, S, mu, Sigma, iter_final_exactw2sq, thin_exactw2sq)$w2sq


###########################
# Do MCMC #################
###########################

# Sample lists of initial values
x0 <- vector(mode = "list", R)
y0 <- vector(mode = "list", R)
for(r in 1:R) {
  x0[[r]] <- as.vector(t_chol_S %*% rnorm(d))
  y0[[r]] <- as.vector(t_chol_S %*% rnorm(d))
}

# Do MCMC in parallel
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)

in_time <- Sys.time()
out <- foreach(i = 1:R, x0_ = x0, y0_ = y0, .packages = "wassersteinbound") %dopar% {
  SetSeed_cpp(seed, i)
  mcmc <- SimulateReflMaxGibbs_ar1(x0_, y0_, std_dev, c, L, iter_store_x, iter_final_x, thin)
  
  return(mcmc)
}

out_time <- Sys.time()
out_time - in_time

parallel::stopCluster(cl)


# Format data for convenience. This will double the memory usage.

source("format_data.R")
rm(a, out)


# Get coupling bounds
tvd_coupling_bound <- AssembleTVDCouplingBound(tau, L, R, thin)
w2_coupling_bound  <- AssembleW2CouplingBound(w2_bound_components, 
                                              tau, L, R, thin)

# Evaluate empirical W2^2, with jackknife leave-one-out estimates

in_time <- Sys.time()
w2sq_out <- EvaluateW2sqJackMCMC(x, x_reference, R, ncores)
out_time <- Sys.time()
out_time - in_time

w2sq      <- w2sq_out$w2sq
jack_w2sq <- w2sq_out$jack_w2sq

#####
# Debias to obtain empirical W2^2 bounds, and compute confidence intervals

source("get_bounds.R")

###########################
# Format data #############
###########################

SignedSquare <- function(x) {
  return(sign(x) * x^2)
}

w2sq_coupling_bound <- w2_coupling_bound^2

df_iter    <- c(seq(0, length(w2sq_coupling_bound) - 1) * thin, 
                seq(0, length(w2sq_ub) - 1) * thin, 
                seq(0, length(w2sq_lb) - 1) * thin, 
                seq(0, length(w2sq_exact) - 1) * thin_exactw2sq)
df_bounds  <- c(w2sq_coupling_bound, 
                w2sq_ub, 
                w2sq_lb, 
                w2sq_exact)
df_which   <- c(rep("w2sq_coupling_bound", length(w2sq_coupling_bound)),
                rep("w2sq_ub", length(w2sq_ub)), 
                rep("w2sq_lb", length(w2sq_lb)), 
                rep("w2sq_exact", length(w2sq_exact)))
df_conf_lo <- c(rep(NA, length(w2sq_coupling_bound)), 
                w2sq_ub - 2 * sqrt(w2sq_ub_jack_var), 
                SignedSquare(w2_lb - 4.5 * sqrt(w2_lb_jack_var)), 
                rep(NA, length(w2sq_exact)))
df_conf_hi <- c(rep(NA, length(w2sq_coupling_bound)), 
                w2sq_ub + 2 * sqrt(w2sq_ub_jack_var), 
                SignedSquare(w2_lb + 4.5 * sqrt(w2_lb_jack_var)), 
                rep(NA, length(w2sq_exact)))

dataf <- data.frame(df_which, df_iter, df_bounds, df_conf_lo, df_conf_hi)
names(dataf) <- c("bound_type", "iter", "bound", "lower_ci", "upper_ci")

#####
# Save output

save(dataf, file = "gibbs.Rdata")

###############
# Plot ########
###############

load(file = "gibbs.Rdata")

library(ggplot2)
library(scales)
library(ggthemes)
# library(gridExtra)

get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


palette <- c("w2sq_coupling_bound" = "#009E73",
             "w2sq_ub" = "#56B4E9",
             "w2sq_lb" = "#E69F00",
             "w2sq_exact" = "#000000")
palette_ribbon <- c("w2sq_coupling_bound" = "transparent",
                    "w2sq_ub" = "#56B4E9",
                    "w2sq_lb" = "#E69F00",
                    "w2sq_exact" = "transparent")
brks <-  c("w2sq_coupling_bound",
           "w2sq_ub",
           "w2sq_lb",
           "w2sq_exact")
labs <- c("Coupling", 
          "Upper (empirical)", 
          "Lower (empirical)", 
          "Exact")

plt <- ggplot(dataf, aes(x = iter, 
                       color = factor(bound_type), fill = factor(bound_type))) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      geom_line(aes(y = bound), 
                size = 1) +
      geom_ribbon(aes(ymin = lower_ci, 
                  ymax = upper_ci), 
                  alpha = 2/10,
                  colour = NA) +
      ylab("Squared Wasserstein distance") +
      xlab("Iteration") +
      scale_colour_manual(values = palette,
                          breaks = brks,
                          labels = labs,
                          name = c("Bound type")) +
      scale_fill_manual(values = palette_ribbon,
                          breaks = brks,
                          labels = labs,
                          name = c("Bound type")) +
      coord_cartesian(xlim = c(0, 2900), ylim = c(0, 3000)) +
      theme_bw() + theme(legend.position="none")
plt

plt_zoom <- plt + 
            coord_cartesian(xlim = c(0, 3000), ylim = c(-5, 50)) +
            ylab(element_blank()) + 
            xlab(element_blank()) + 
            theme(plot.background = element_rect(colour = "black"))
plt_zoom

plt <- plt + 
  annotation_custom(
    ggplotGrob(plt_zoom), 
    xmin = 1000, xmax = 3000, ymin = 500, ymax = 3000
  )
plt

# Save plot
ggsave(filename = "gibbs.pdf",
       plot = plt,
       width = 16, height = 12, units = "cm")









