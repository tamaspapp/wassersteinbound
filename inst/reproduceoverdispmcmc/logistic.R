library(wassersteinbound)
library(ggplot2)
library(ars)

DrawFromLogistic <- function(n, d) {
  if(d == 1){
    radials <- rlogis(n)
  } else {
    # Adaptive rejection sampling to sample radial component
    log_radial <- function(r) {
      (d - 1) * log(r) - r - 2 * log(1 + exp(-r))
    }
    log_radial_deriv <- function(r) {
      (d - 1) / r - 1 + 2 * exp(-r) / (1 + exp(-r))
    }
    
    log_radial <- function(r) {
      (d - 1) * log(r) - r - 2 * log1p(exp(-r))
    }
    log_radial_deriv <- function(r) {
      (d - 1) / r + 1 - 2 / (1 + exp(-r))
    }
    
    radials <- ars::ars(n, f = log_radial, fprima = log_radial_deriv, x = c(0.1, 1, d - 1, 11, 2*d), m = 5, lb = TRUE, xlb = 0)
  }
  
  draws <- vector(mode = "list", length = n)
  
  norms <- rnorm(n * d)
  for(i in 1:n){
    v <- norms[1:d + (i-1)*d]
    unif <- v / sqrt(sum(v^2))
    draws[[i]] <- radials[i] * unif
  }
  
  return(draws)
}


# Randomness
seed <- 12345
SetSeed_cpp(seed)
set.seed(seed)


############
# d = 1
############

# MCMC parameters ##
d    <- 1L   # Dimension
iter <- 20L  # Iteration count 
R    <- 1e6L # Replicate count
h    <- 3    # Step size

x0 <- DrawFromLogistic(R, d)
for(i in 1:R) {
  x0[[i]] <- 2 * x0[[i]]
}

acc_rate <- mean(RWMLogistic(x0[[1]], h, 1e5L)$acc)
acc_rate


out <- matrix(NA, nrow = R, ncol = iter + 1)
in_time <- Sys.time()
for(i in 1:R) {
  out[i, ] <-  RWMLogistic(x0[[i]], h, iter)$x_norms
}
out_time <- Sys.time()
out_time - in_time



#####
# Compute and plot quantiles
#####

thin <- 1L

prob <- seq(0.005, 0.995, length.out = 20)
iters_plot <- seq.int(0, iter, thin)

num_prob <- length(prob)
iters <- rep(iters_plot, each = num_prob)
probs <- rep(prob, length(iters_plot))
quants <- NULL
for(i in iters_plot) {
  quants <- c(quants, quantile(out[, i + 1], probs = prob))
  # quants <- c(quants, quantile(out[, i + 1] * ifelse(runif(R) < 0.5, -1, 1), probs = prob))
}

df <- data.frame(cbind(quants, iters, probs))
names(df) <- c("quantile", "iteration", "probability")

d1 <- ggplot(data = df, aes(x = iteration, y = quantile, color = probability, group = probability)) +
  geom_line() +
  xlab("Iteration") +
  ylab("Quantile") +
  theme_bw() + 
  theme(legend.position = "none")
d1 

ggsave(filename = "logistic_d=1.pdf",
       plot = d1,
       width = 8, height = 8, units = "cm")

############
# d = 10
############

# MCMC parameters ##
d    <- 10L   # Dimension
iter <- 300L  # Iteration count 
R    <- 1e6L  # Replicate count
h    <- 2.5   # Step size

x0 <- DrawFromLogistic(R, d)
for(i in 1:R) {
  x0[[i]] <- 2 * x0[[i]]
}

acc_rate <- mean(RWMLogistic(x0[[1]], h, 1e5L)$acc)
acc_rate


out <- matrix(NA, nrow = R, ncol = iter + 1)
in_time <- Sys.time()
for(i in 1:R) {
  out[i, ] <-  RWMLogistic(x0[[i]], h, iter)$x_norms
}
out_time <- Sys.time()
out_time - in_time



#####
# Compute and plot quantiles
#####

thin <- 1L

prob <- seq(0.005, 0.995, length.out = 20)
iters_plot <- seq.int(0, iter, thin)

num_prob <- length(prob)
iters <- rep(iters_plot, each = num_prob)
probs <- rep(prob, length(iters_plot))
quants <- NULL
for(i in iters_plot) {
  quants <- c(quants, quantile(out[, i + 1], probs = prob))
  # quants <- c(quants, quantile(out[, i + 1] * ifelse(runif(R) < 0.5, -1, 1), probs = prob))
}

df <- data.frame(cbind(quants, iters, probs))
names(df) <- c("quantile", "iteration", "probability")

d10 <- ggplot(data = df, aes(x = iteration, y = quantile, color = probability, group = probability)) +
  geom_line() +
  xlab("Iteration") +
  ylab("Quantile") +
  theme_bw() + 
  theme(legend.position = "none")
d10


ggsave(filename = "logistic_d=10.pdf",
       plot = d10,
       width = 8, height = 8, units = "cm")




