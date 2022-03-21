
source("svm_params.R")

# Step size
h <- 0.25 / t^(1/2)

# Iterations for X-chain storage
iter_final_x <- 3e6L
iter_store_x <- as.integer(2e6L)

# Thinning factor
thin <- 2e3L

# Which iterations the bound is computed for
which_iters <- seq(0, iter_store_x, thin)

# Number of cores to be used
ncores <- parallel::detectCores()

# Debiasing for the bound
start_debias <- 1e6L
end_debias <- iter_store_x

debias_iters <- seq(floor(start_debias/ thin) + 1, floor(end_debias/ thin) + 1)

# Do MCMC in parallel
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)

in_time <- Sys.time()
out <- foreach(i = 1:R, .packages = "wassersteinbound") %dopar% {
  SetSeed_cpp(seed, i)
  x0 <- SampleLatentVariables(t, sig, phi)
  SimulateRWM_SVM(x0, y, beta, sig, phi, h, iter_store_x, iter_final_x, thin)
}

out_time <- Sys.time()
print(out_time - in_time)

parallel::stopCluster(cl)


# Format data for convenience. This will double the memory usage.
x <- vector(mode = "list", length = floor(iter_store_x / thin) + 1)

a <- matrix(NA, nrow = t, ncol = R)

for (i in 1:length(x)) {
  
  for(j in 1:R) {
    a[, j] <- out[[j]]$x[[i]]
  }
  
  x[[i]] <- a
}

x_reference <- matrix(NA, nrow = t, ncol = R)

for (j in 1:R) {
  x_reference[, j] <- out[[j]]$x_final
}

rm(a, out)


# Evaluate empirical W2^2, with jackknife leave-one-out estimates
w2sq_out <- EvaluateW2sqJackMCMC(x, x_reference, R, ncores)

w2sq      <- w2sq_out$w2sq
jack_w2sq <- w2sq_out$jack_w2sq

plot(w2sq)

#####
# Debias to obtain empirical W2^2 bounds, and compute confidence intervals

source("get_bounds.R")


#####
# Save output

save(w2sq, jack_w2sq, w2sq_ub, w2sq_ub_jack_var, 
     w2sq_lb, w2sq_lb_jack_var,
     w2_lb, w2_lb_jack_var,
     file = "rwm.Rdata")


###########################
# RWM #####################
###########################  
library(ggplot2)
library(scales)
library(ggthemes)

df_iter    <- c(seq(0, length(w2sq_ub) - 1) * thin, 
                seq(0, length(w2sq_lb) - 1) * thin)
df_bounds  <- c(w2sq_ub, 
                w2sq_lb)
df_which   <- c(rep("w2sq_ub", length(w2sq_ub)), 
                rep("w2sq_lb", length(w2sq_lb)))
df_conf_lo <- c(w2sq_ub - 2 * sqrt(w2sq_ub_jack_var), 
                SignedSquare(w2_lb - 4.5 * sqrt(w2_lb_jack_var)))
df_conf_hi <- c(w2sq_ub + 2 * sqrt(w2sq_ub_jack_var), 
                SignedSquare(w2_lb + 4.5 * sqrt(w2_lb_jack_var)))

df_rwm <- data.frame(df_which, df_iter, df_bounds, df_conf_lo, df_conf_hi)
names(df_rwm) <- c("bound_type", "iter", "bound", "lower_ci", "upper_ci")



plt_rwm <- ggplot(df_rwm, aes(x = iter, 
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
  coord_cartesian(xlim = c(0, 1e6L), ylim = c(0, 150)) +
  theme_bw() + theme(legend.position="none")
plt_rwm

plt_rwm_zoom <- plt_rwm + 
  coord_cartesian(xlim = c(0, 1e6L), ylim = c(-0.5, 3)) + 
  theme(legend.position="none") + 
  ylab(element_blank()) + 
  xlab(element_blank()) + 
  theme(plot.background = element_rect(colour = "black"))
plt_rwm_zoom

plt_rwm <- plt_rwm + 
  annotation_custom(
    ggplotGrob(plt_rwm_zoom), 
    xmin = 2.5e5L, xmax = 1e6L, ymin = 25, ymax = 150
  )

plt_rwm

ggsave(filename = "svm_optimal_scaling_rwm.pdf",
       plot = plt_rwm,
       width = 16, height = 12, units = "cm")
