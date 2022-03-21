library(ggplot2)
library(scales)
library(reshape2)
######
# Compute lower bound and mixing times

GetMixingTimes <- function(dat, iter, thin, D) {
  mats <- vector("list", length(dat))
  
  for(i in 1:length(dat)) {
    out <- dat[[i]]
    
    # Coupling mixing time upper bounds
    tvd_ub_025_coupling <- min(which(out$tvd_coupling_bound < 0.25)) - 1
    w2sq_ub_6_coupling  <- min(which(out$w2_coupling_bound^2 < 6)) - 1
      
    # Empirical mixing time upper and lower bounds
    w2sq_ub_6_empirical <- (min(which(out$w2sq_ub < 6)) - 1) * thin
    w2sq_lb_6_empirical <- max((max(which(out$w2_lb^2 >= 6)) - 1) * thin, 0)
    
    mixing_times <- rep(NA, 5)
    
    mixing_times[1] <- D[i]
    mixing_times[2] <- tvd_ub_025_coupling
    mixing_times[3] <- w2sq_ub_6_coupling
    mixing_times[4] <- w2sq_ub_6_empirical
    mixing_times[5] <- w2sq_lb_6_empirical
    
    
    mats[[i]] <- mixing_times
  }
  
  mixing_times <- do.call(rbind, mats)
  mixing_times <- as.data.frame(mixing_times)
  
  names(mixing_times) <- c("d", "tvd_ub_025_coupling", "w2sq_ub_6_coupling",
                           "w2sq_ub_6_empirical", "w2sq_lb_6_empirical")
  
  return(mixing_times)
}

source("params.R")

####
# MALA
####

H <- 1 / D^(1/6)

disp <- sqrt(3)

start_debias <- 250L
iter         <- 500L
L            <- 2000L
iter_final   <- L
thin         <- 1L

thin_cpl <- 1L

end_debias   <- iter
debias_iters <- seq(floor(start_debias/ thin) + 1, floor(end_debias/ thin) + 1)

load("mala.Rdata")

len <- length(D)
mala_mix <- GetMixingTimes(mala, iter, thin, D)
mala_mix 
mala_mix[["mcmc"]] <- "mala"
mala_mix [["tvd_ub_025_coupling"]] <- NULL

####
# ULA
####

H <- 0.2 / D^(1/4)
disp <- sqrt(3)

start_debias <- 8000L
iter         <- 20000L
L            <- 70000L
iter_final   <- L
thin         <- 40L

thin_cpl <- 1L

end_debias   <- iter
debias_iters <- seq(floor(start_debias/ thin) + 1, floor(end_debias/ thin) + 1)

load("ula.Rdata")

len <- length(D)
ula_mix <- GetMixingTimes(ula, iter, thin, D)
ula_mix
ula_mix[["mcmc"]] <- "ula"
ula_mix[["tvd_ub_025_coupling"]] <- NULL

load("ula_exact_mixing.Rdata")
ula_mix[["w2sq_6_exact"]] <- w2sq_mix_true
# mala_mix[["w2sq_6_exact"]] <- NA

######
# Reshape data into tall format

mixing_times_mala <- reshape2::melt(mala_mix , id = c("d", "mcmc"))
mixing_times_ula  <- reshape2::melt(ula_mix, id = c("d", "mcmc"))

# Combine data frames into one
dat <- rbind(mixing_times_mala, mixing_times_ula)
dat <- subset(dat, select = -c(R))
######

######
# Plot the mixing time scaling

palette <- c("w2sq_ub_6_coupling" = "#009E73",
             "w2sq_ub_6_empirical" = "#56B4E9",
             "w2sq_lb_6_empirical" = "#E69F00",
             "w2sq_6_exact" = "#000000")

labs <- c("Coupling", "Upper (empirical)", "Lower (empirical)", "Exact")

plt <- ggplot(dat, aes(x = d, 
                color = factor(variable),
                group = interaction(mcmc, variable))) +
  stat_summary(aes(y = value, 
                   linetype = factor(mcmc)), 
               fun = mean, 
               geom = "line", 
               size = 1)+
  geom_point(aes(y = value,
                 shape = factor(variable)), 
             size = 2.5) +
  scale_x_continuous(trans = "log10",
                     breaks = D, 
                     minor_breaks = c()) +
  scale_y_continuous(trans = "log10",
                     breaks = c(2, 6, 20, 60, 200, 600, 2000, 6000, 20000),
                     minor_breaks = c(),
                     limits = c(0.9, 25000)) +
  ylab("Mixing time") +
  xlab("Dimension") +
  #ggtitle("Dimensional scaling of mixing time: W2 < sqrt(6)") +
  scale_colour_manual(values = palette,
                      breaks = c("w2sq_ub_6_coupling",
                                 "w2sq_ub_6_empirical",
                                 "w2sq_lb_6_empirical",
                                 "w2sq_6_exact"),
                      labels = labs,
                      name = c("Bound type")) +
  scale_shape_discrete(solid = TRUE, 
                       labels = labs,
                       name = c("Bound type")) +
  scale_linetype_discrete(labels = c("MALA", "ULA"), 
                          name = "MCMC algorithm") +
  #geom_function(fun = transf, color = "black") +
  theme_bw() +
  theme(legend.position = "none")

ggsave(filename = "mala_vs_ula.pdf",
       plot = plt,
       width = 16, height = 12, units = "cm")

