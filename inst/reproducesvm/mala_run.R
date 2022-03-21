# Setup
source("svm_params.R")

# Step size
h <- 0.145 / t^(1/6)

# Iterations for X-chain storage
iter_final_x <- 6e4L
iter_store_x <- 5e4L

# Lag
L <- iter_final_x

# Thinning factor
thin <- as.integer(iter_store_x / 1000)

# Which iterations the bound is computed for
which_iters <- seq(0, iter_store_x, thin)

# Number of cores to be used
ncores <- parallel::detectCores()

# Debiasing for the bound
start_debias <- 2e4L
end_debias   <- iter_store_x
debias_iters <- seq(floor(start_debias/ thin) + 1, floor(end_debias/ thin) + 1)


# Do MCMC in parallel
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)

in_time <- Sys.time()
out <- foreach(i = 1:R, .packages = "wassersteinbound") %dopar% {
  SetSeed_cpp(seed, i)
  x0 <- SampleLatentVariables(t, sig, phi)
  y0 <- SampleLatentVariables(t, sig, phi)
  mcmc <- SimulateReflMaxMALA_SVM(x0, y0, y, beta, sig, phi, h, L, iter_store_x, iter_final_x, thin)

  return(mcmc)
}

out_time <- Sys.time()
out_time - in_time

parallel::stopCluster(cl)


# Format data for convenience. This will double the memory usage.
source("format_data.R")


# Save samples from reference measure as target samples
target_sample <- x_reference
save(target_sample, file = "svm_sample.RData")

# Get coupling bounds
tvd_coupling_bound <- AssembleTVDCouplingBound(tau, L, R, thin)
w2_coupling_bound  <- AssembleW2CouplingBound(w2_bound_components, 
                                              tau, L, R, thin)

# Evaluate empirical W2^2, with jackknife leave-one-out estimates

w2sq_out <- EvaluateW2sqJackMCMC(x, x_reference, R, ncores)

w2sq      <- w2sq_out$w2sq
jack_w2sq <- w2sq_out$jack_w2sq


#####
# Debias to obtain empirical W2^2 bounds, and compute confidence intervals

source("get_bounds.R")


#####
# Save output

save(w2sq, jack_w2sq, w2sq_ub, w2sq_ub_jack_var, 
     w2sq_lb, w2sq_lb_jack_var,
     w2_lb, w2_lb_jack_var,
     w2_bound_components,
     tau, tvd_coupling_bound, w2_coupling_bound, 
     file = "mala.Rdata")



###########################
# MALA ####################
###########################

which_iters_coupling <- seq(0, max(tau - L) - 1, thin)

# Format data #############

SignedSquare <- function(x) {
  return(sign(x) * x^2)
}

w2sq_coupling_bound <- w2_coupling_bound^2

df_iter    <- c(seq(0, length(w2sq_coupling_bound) - 1) * thin, 
                seq(0, length(w2sq_ub) - 1) * thin, 
                seq(0, length(w2sq_lb) - 1) * thin)
df_bounds  <- c(w2sq_coupling_bound, 
                w2sq_ub, 
                w2sq_lb)
df_which   <- c(rep("w2sq_coupling_bound", length(w2sq_coupling_bound)),
                rep("w2sq_ub", length(w2sq_ub)), 
                rep("w2sq_lb", length(w2sq_lb)))
df_conf_lo <- c(rep(NA, length(w2sq_coupling_bound)), 
                w2sq_ub - 2 * sqrt(w2sq_ub_jack_var), 
                SignedSquare(w2_lb - 4.5 * sqrt(w2_lb_jack_var)))
df_conf_hi <- c(rep(NA, length(w2sq_coupling_bound)), 
                w2sq_ub + 2 * sqrt(w2sq_ub_jack_var), 
                SignedSquare(w2_lb + 4.5 * sqrt(w2_lb_jack_var)))

df_mala <- data.frame(df_which, df_iter, df_bounds, df_conf_lo, df_conf_hi)
names(df_mala) <- c("bound_type", "iter", "bound", "lower_ci", "upper_ci")

# Plot ########

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

plt_mala <- ggplot(df_mala, aes(x = iter, 
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
  coord_cartesian(xlim = c(0, 40000), ylim = c(0, 300)) +
  theme_bw() + theme(legend.position="none")
# plt_mala

plt_mala_zoom <- plt_mala + 
  coord_cartesian(xlim = c(0, 40000), ylim = c(-0.5, 3)) + 
  theme(legend.position="none") + 
  ylab(element_blank()) + 
  xlab(element_blank()) + 
  theme(plot.background = element_rect(colour = "black"))
plt_mala_zoom

plt_mala <- plt_mala + 
  annotation_custom(
    ggplotGrob(plt_mala_zoom), 
    xmin = 10000, xmax = 40000, ymin = 50, ymax = 300
  )
plt_mala

ggsave(filename = "svm_optimal_scaling_mala.pdf",
       plot = plt_mala,
       width = 16, height = 12, units = "cm")
