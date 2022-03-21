
library(wassersteinbound)
library(doParallel)

GetExactW2sq <- function(d, sigma_sq) {return(d * (sqrt(sigma_sq) - 1) ^ 2)}

GetEmpiricalW2sqBounds <- function(d, sigma_sq, n, reps, ncores) {
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  SquareSign <- function(x) { return(sign(x) * x^2)}
  
  out <- foreach(j = 1:reps, .packages = c("wassersteinbound"), .combine = c) %dopar% {
    
    x <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = d, ncol = n)
    y <- matrix(rnorm(n * d), nrow = d, ncol = n)
    z <- matrix(rnorm(n * d), nrow = d, ncol = n)
    
    first_term  <- SolveAssignmentNetworkflow(EvaluateSquaredCost(x, z))
    second_term <- SolveAssignmentNetworkflow(EvaluateSquaredCost(y, z))
    
    w2sq_ub_ <- first_term - second_term
    w2sq_lb_ <- SquareSign(sqrt(first_term) - sqrt(second_term))
    
    a <- c(first_term, w2sq_ub_, w2sq_lb_)
    
    return(a)
  }
  
  w2sq_empirical <- c(w2sq_empirical, out[c(TRUE, FALSE, FALSE)]) 
  w2sq_ub <- c(w2sq_ub, out[c(FALSE, TRUE, FALSE)]) 
  w2sq_lb <- c(w2sq_lb, out[c(FALSE, FALSE, TRUE)])
  
  parallel::stopCluster(cl)
  
  data_frame <- data.frame(rep(d, reps), rep(sigma_sq, reps), rep(n, reps), w2sq_empirical, w2sq_ub, w2sq_lb)
  names(data_frame) <- c("d", "sigma_sq", "n", "w2sq_empirical", "w2sq_ub", "w2sq_lb")
  return(data_frame)
}


seed <- 12345
ncores <- 4
reps <- 100

D <- c(10, 100)
N <- c(10, 100, 1000)
Sigma_sq <- c(1.1, 2, 10)

w2sq_exact <- data.frame(matrix(ncol = 3, nrow = 0))
w2sq_bounds <- data.frame(matrix(ncol = 6, nrow = 0))

set.seed(seed)

for (sigma_sq in Sigma_sq) {
  for(d in D) {
    w2sq_exact <- rbind(w2sq_exact, c(d, sigma_sq, GetExactW2sq(d, sigma_sq)))
    for(n in N) {
      w2sq_bounds <- rbind(w2sq_bounds, GetEmpiricalW2sqBounds(d, sigma_sq, n, reps, ncores))
    }
  }
}

names(w2sq_exact) <- c("d", "sigma_sq", "w2sq_exact")
names(w2sq_bounds) <- c("d", "sigma_sq", "n", "w2sq_empirical", "w2sq_ub", "w2sq_lb")

save(w2sq_exact, w2sq_bounds, file = "prelim.Rdata")

######
# Convert data to long format and plot

load("prelim.Rdata")

library(ggplot2)
library(reshape2)
library(latex2exp)

palette <- c("w2sq_empirical" = "#CC79A7",
             "w2sq_ub" = "#56B4E9",
             "w2sq_lb" = "#E69F00",
             "w2sq_exact" = "#000000")
brks <-  c("w2sq_empirical",
           "w2sq_ub",
           "w2sq_lb",
           "w2sq_exact")
labels <- c("Plug-in",
            "Upper", 
            "Lower", 
            "Exact")


w2sq_bounds_long <- reshape2::melt(w2sq_bounds, id = c("d", "sigma_sq", "n"))

# Plot

# One plot for each (d, sigma_sq) combination
plt <- vector(mode = "list", length = length(Sigma_sq) * length(D))
exact <- rep(NA, length(Sigma_sq) * length(D))
avg_rel_size_lb <- rep(NA, length(Sigma_sq) * length(D))

k <- 0
for(d in D) {
  for (sigma_sq in Sigma_sq) {
    k <- k + 1
    w2sq_bounds_ <- w2sq_bounds_long[w2sq_bounds_long$d == d & w2sq_bounds_long$sigma_sq == sigma_sq, ]
    w2sq_exact_ <- w2sq_exact[w2sq_exact$d == d & w2sq_exact$sigma_sq == sigma_sq, ]
  
    exact[k] <- w2sq_exact_$w2sq_exact
    
    plt[[k]] <- ggplot(data = w2sq_bounds_,aes(x = factor(n), y = value)) +
      geom_boxplot(aes(fill = factor(variable))) +
      ylab("Squared Waserstein distance") +
      xlab("Sample size") +
      # ggtitle(TeX(paste0("(",letters[k],") ","$d = ",d,"$,","$\\sigma^2 = ",sigma_sq,"$"))) +
      scale_fill_manual(values = palette,
                        breaks = brks,
                        labels = labels,
                        name = c("Estimator")) +
      scale_color_manual(values = palette,
                         breaks = brks,
                         labels = labels,
                         name = element_blank()) +
      geom_hline(yintercept = exact[k],linetype = "dashed") +
      theme_bw() + theme(legend.position = "none")
    print(d)
    print(sigma_sq)
    avg_rel_size_lb[k] <- mean(w2sq_bounds[w2sq_bounds$d == d & w2sq_bounds$sigma_sq == sigma_sq & w2sq_bounds$n == 1000, ]$w2sq_lb) / exact[k]
    
    }
}
avg_rel_size_lb

plt[[1]]

# Save plots
ggsave(filename = "prelim_a.pdf",
       plot = plt[[1]],
       width = 10, height = 10, units = "cm")
ggsave(filename = "prelim_b.pdf",
       plot = plt[[2]],
       width = 10, height = 10, units = "cm")
ggsave(filename = "prelim_c.pdf",
       plot = plt[[3]],
       width = 10, height = 10, units = "cm")
ggsave(filename = "prelim_d.pdf",
       plot = plt[[4]],
       width = 10, height = 10, units = "cm")
ggsave(filename = "prelim_e.pdf",
       plot = plt[[5]],
       width = 10, height = 10, units = "cm")
ggsave(filename = "prelim_f.pdf",
       plot = plt[[6]],
       width = 10, height = 10, units = "cm")
