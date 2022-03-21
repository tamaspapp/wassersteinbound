
library(wassersteinbound)
library(doParallel)

# Function below assumes that d is even
GetExactW2sq <- function(d, sigma_sq) {
  return( d/2 * ( (sqrt(sigma_sq) - 1) ^ 2 + (sqrt(sigma_sq) - 2) ^ 2))
}

GetEmpiricalW2sqBounds <- function(d, sigma_sq, n, reps, ncores) {
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  w2sq_ub <- foreach(j = 1:reps, .packages = c("wassersteinbound"), .combine = c) %dopar% {
    
    x <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = d, ncol = n)
    y <- matrix(rnorm(n * d, sd = c(1,2)), nrow = d, ncol = n)
    z <- matrix(rnorm(n * d, sd = c(1,2)), nrow = d, ncol = n)
    
    first_term  <- SolveAssignmentNetworkflow(EvaluateSquaredCost(x, z))
    second_term <- SolveAssignmentNetworkflow(EvaluateSquaredCost(y, z))
    
    w2sq_ub_ <- first_term - second_term
    
    return(w2sq_ub_)
  }
  
  parallel::stopCluster(cl)
  
  data_frame <- data.frame(rep(d, reps), rep(sigma_sq, reps), rep(n, reps), w2sq_ub)
  names(data_frame) <- c("d", "sigma_sq", "n", "w2sq_ub")
  return(data_frame)
}


seed <- 12345
ncores <- parallel::detectCores()
reps <- 100

D <- c(10, 100)
N <- c(10, 100, 1000)
Sigma_sq <- seq(1, 4, length = 5)

w2sq_exact <- data.frame(matrix(ncol = 3, nrow = 0))
w2sq_bound <- data.frame(matrix(ncol = 4, nrow = 0))

set.seed(seed)

for (sigma_sq in Sigma_sq) {
  for(d in D) {
    w2sq_exact <- rbind(w2sq_exact, c(d, sigma_sq, GetExactW2sq(d, sigma_sq)))
    for(n in N) {
      w2sq_bound <- rbind(w2sq_bound, GetEmpiricalW2sqBounds(d, sigma_sq, n, reps, ncores))
    }
  }
}
names(w2sq_exact) <- c("d", "sigma_sq", "w2sq_exact")
names(w2sq_bound) <- c("d", "sigma_sq", "n", "w2sq_ub")

save(w2sq_exact, w2sq_bound, file = "overdisp.Rdata")

######
# Convert data to long format and plot

load("overdisp.Rdata")

library(ggplot2)
library(reshape2)
library(latex2exp)

w2sq_bound <- merge(w2sq_bound, w2sq_exact, by = c("d", "sigma_sq"))

bound_diff <- w2sq_bound[,c(1,2,3,5)]
bound_diff$"bound_diffs" <- w2sq_bound$w2sq_ub - w2sq_bound$w2sq_exact

# Plot ###
#
# One plot for each d

palette <- c("thousand" = "#CC79A7",
             "hundred" = "#56B4E9",
             "ten" = "#E69F00")
brks <-  factor(N)
labels <- c("n = 10",
            "n = 100", 
            "n = 1000")


plt <- vector(mode = "list", length = length(D))

k <- 0
  for(d in D) {
    k <- k + 1
    
    bound_diff_ <- bound_diff[bound_diff$d == d, ]
    
    plt[[k]] <- ggplot(data = bound_diff_, aes(x = factor(sigma_sq), y = bound_diffs / w2sq_exact)) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      geom_boxplot(aes(fill = factor(n))) +
      ylab("Relative error") +
      xlab(TeX("$\\sigma^2$")) +
      #ggtitle(TeX(paste0("(",letters[k],") ","$d = ",d))) +
      scale_fill_manual(values = c("#CC79A7",
                                   "#56B4E9",
                                   "#E69F00"),
                        labels = labels,
                        name = c("Sample size")) +
      theme_bw() + theme(legend.position = "none")
  }


# Save plot
ggsave(filename = "overdisp_a.pdf",
       plot = plt[[1]],
       width = 12, height = 8, units = "cm")
ggsave(filename = "overdisp_b.pdf",
       plot = plt[[2]],
       width = 12, height = 8, units = "cm")