library(wassersteinbound)
library(doParallel)

GetEmpiricalW2sqBoundsJack <- function(D, Sigma_sq, N, reps, ncores) {
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  # Data frame with columns "d", "sigma_sq", "n", "w2sq_empirical", "w2sq_ub", "w2_lb", "w2sq_empirical_jackvar", "w2sq_ub_jackvar", "w2_lb_jackvar"
  data_frame <- foreach(j = 1:reps, 
                        .packages = c("wassersteinbound", "foreach"), 
                        .combine = "rbind", .inorder = FALSE) %dopar% {
    foreach(d = D, .combine = "rbind") %do% {
      foreach(sigma_sq = Sigma_sq, .combine = "rbind") %do% {
        foreach(n = N, .combine = "rbind") %do% {
          
          GetJackknifeVariance <- function(loo_estimates) {
            sample_size <- length(loo_estimates)
            return((sample_size - 1) / sample_size * sum((loo_estimates - mean(loo_estimates))^2))
          }
          
          x <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = d, ncol = n)
          y <- matrix(rnorm(n * d), nrow = d, ncol = n)
          z <- matrix(rnorm(n * d), nrow = d, ncol = n)
          
          t1 <- Flapjack(EvaluateSquaredCost(x, z))
          t2 <- Flapjack(EvaluateSquaredCost(y, z))
          
          # Transportation costs
          c1 <- t1$transp_cost
          c2 <- t2$transp_cost
          
          # Leave-one-out transportation costs
          j1 <- t1$jack_data
          j2 <- t2$jack_data
          
          return(c(d, sigma_sq, n,
                   c1,
                   GetJackknifeVariance(j1),
                   c1 - c2,
                   GetJackknifeVariance(j1 - j2),
                   sqrt(c1) - sqrt(c2),
                   GetJackknifeVariance(sqrt(j1) - sqrt(j2))))
          return(c(c1, j1))
        }
      }
    }
  }
  
  parallel::stopCluster(cl)
  
  data_frame <- data.frame(data_frame)
  names(data_frame) <- c("d", "sigma_sq", "n",
                         "w2sq_empirical", "w2sq_empirical_jackvar",
                         "w2sq_ub", "w2sq_ub_jackvar",
                         "w2_lb", "w2_lb_jackvar")
  return(data_frame)
}

GetBootstrapCIs <- function(jack_df, D, Sigma_sq, N, boot_reps){
  
  out <- foreach(d = D, .combine = "rbind") %do% {
    foreach(sigma_sq = Sigma_sq, .combine = "rbind") %do% {
      foreach(n = N, .combine = "rbind") %do% {
        
        data_frame <- jack_df[jack_df$d == d & jack_df$sigma_sq == sigma_sq & jack_df$n == n, ]
        
        return(c("d" = d,
                 "sigma_sq" = sigma_sq,
                 "n" = n,
                 "w2sq_empirical_var" = var(data_frame$w2sq_empirical),
                 "w2sq_empirical_var_bootsd" = sd(BoostrapVarianceBasic(data_frame$w2sq_empirical, boot_reps)), 
                 "w2sq_ub_var" = var(data_frame$w2sq_ub),
                 "w2sq_ub_var_bootsd" = sd(BoostrapVarianceBasic(data_frame$w2sq_ub, boot_reps)),
                 "w2_lb_var" = var(data_frame$"w2_lb"),
                 "w2_lb_var_bootsd"= sd(BoostrapVarianceBasic(data_frame$w2_lb, boot_reps))
        ))
      }
    }
  }
  return(out)
}

seed <- 12345
ncores <- parallel::detectCores()
reps <- 500
boot_reps <- reps

D <- c(10, 100)
N <- c(10, 100, 1000)
Sigma_sq <- c(1.1, 10)

set.seed(seed)
jack_df <- GetEmpiricalW2sqBoundsJack(D, Sigma_sq, N, reps, ncores)
boot_df <- data.frame(GetBootstrapCIs(jack_df, D, Sigma_sq, N, boot_reps))

save(jack_df, boot_df, file = "jack.RData")


###
# Boostrap variance estimate for the sample variance estimator

load("jack.Rdata")


######
# Plot

library(ggplot2)
library(scales)
# library(latex2exp)
# library(lemon)

get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

# Set custom plot parameters
x_shift  <- 0.25
width    <- 0.4
x_breaks <- c(10, 100, 1000)


plt <- vector(mode = "list", length = length(Sigma_sq) * length(D))

k <- 0
for (sigma_sq in Sigma_sq) {
  for(d in D) {
    
    plot_df     <- jack_df[, c(1, 2, 3, 6, 7)][jack_df$"d" == d & jack_df$"sigma_sq" == sigma_sq, ]
    w2sq_ub_var <- boot_df[, c(1, 2, 3, 6, 7)][boot_df$"d" == d & boot_df$"sigma_sq" == sigma_sq, ]
    
    k <- k + 1
    plt[[k]] <- ggplot(plot_df, aes(x = factor(n), group = n)) +
      geom_boxplot(aes(y = w2sq_ub_jackvar, fill = "jackknife"), width = width,
                   position = position_nudge(x = x_shift)) +
                   #position = position_dodge2())+
      geom_point(data = w2sq_ub_var,
                 aes(y = w2sq_ub_var, color = "empirical"),
                 position = position_nudge(x = -x_shift),
                 size = 2) +
                 #position = position_dodge2())+
      geom_errorbar(data = w2sq_ub_var,
                    aes(ymin = w2sq_ub_var - 2 * w2sq_ub_var_bootsd,
                        ymax = w2sq_ub_var + 2 * w2sq_ub_var_bootsd,
                        color = "empirical"),
                    width = width,
                    position = position_nudge(x = -x_shift)) +
      # scale_x_continuous(trans = "log10",
      #                    breaks = x_breaks,
      #                    minor_breaks = c(), expand = c(1,0)) +
      scale_color_manual(values = c("empirical" = "red"),
                         labels = c("Empirical (95% CI)"),
                         name = element_blank()) +
      scale_fill_manual(values = c("jackknife" = "#56B4E9"),
                        labels = c("Jackknife"),
                        name = element_blank()) +
      ## This transforms the y-axis AFTER computing all boxplot statistics.
      coord_trans(y = "log10") +
      scale_y_continuous(breaks = breaks_log(), minor_breaks = c()) +
      xlab("Sample size") +
      ylab("Variance") + 
      theme_bw() + theme(legend.position = "none")
  }
}

plt[[1]]

# Get coverage of 95% jackknife confidence intervals

jack_ci_95_coverage <- data.frame(matrix(ncol = 4, nrow = 0))

for (sigma_sq in Sigma_sq) {
  for(d in D) {
    for (n in N) {
      coverage_df <- jack_df[, c(1, 2, 3, 6, 7)][jack_df$d == d & jack_df$n == n & jack_df$sigma_sq == sigma_sq, ]
      
      upper <- coverage_df$w2sq_ub + 2 * sqrt(coverage_df$w2sq_ub_jackvar)
      lower <- coverage_df$w2sq_ub - 2 * sqrt(coverage_df$w2sq_ub_jackvar)
      avg   <- mean(coverage_df$w2sq_ub)

      jack_ci_95_coverage <- rbind(jack_ci_95_coverage,
                                   c(d, sigma_sq, n, mean( (upper > avg) & (lower < avg) ))
                                   )
    }
  }
}

names(jack_ci_95_coverage) <- c("d", "sigma_sq", "n", "coverage")
print(jack_ci_95_coverage)

ggsave(filename = "jack_a.pdf", 
       plot = plt[[1]],
       width = 6, height = 8, units = "cm")
ggsave(filename = "jack_b.pdf", 
       plot = plt[[2]],
       width = 12, height = 8, units = "cm")
ggsave(filename = "jack_c.pdf", 
       plot = plt[[3]],
       width = 12, height = 8, units = "cm")
ggsave(filename = "jack_d.pdf", 
       plot = plt[[4]],
       width = 12, height = 8, units = "cm")
