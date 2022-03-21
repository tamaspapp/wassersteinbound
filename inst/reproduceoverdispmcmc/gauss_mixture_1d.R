library(wassersteinbound)
library(ggplot2)

DrawFromGaussianMixture <- function(n, p, mu, sig) {
  K <- length(p)
  k <- sample.int(K, n, replace = TRUE, prob = p)
  draws <- rnorm(n, mu[k], sig[k])
  return(draws)
}

#####
# Well-separated mixture ####
#####

# Gaussian mixture target parameters
p  <- c(0.5, 0.5) # proportions p
mu <- c(-5, 5)    # means mu
sig <- c(1, 1)    # Standard deviations sigma

# Plot the target
eval_pdf <- seq(-10, 10, by = 0.05)
pdf_vals <- GaussianMixtureDensity(eval_pdf, p, mu, sig)
qplot(eval_pdf, pdf_vals,
      geom = "line",
      xlab = "x",
      ylab = "Target density at x") + theme_bw()

# Randomness
seed <- 12345
SetSeed_cpp(seed)
set.seed(seed)


#####
# Diffusive RWM, small step size
#####

# MCMC parameters ##
iter <- 30L    # Iteration count 
R    <- 1e6L   # Replicate count
h    <- 2      # Step size

x0 <- DrawFromGaussianMixture(R, p, 2 * mu, sig)
out <- matrix(NA, nrow = R, ncol = iter + 1)

in_time <- Sys.time()

for(i in 1:R) {
  out[i, ] <-  RWMMixture1d(x0[i], p, mu, sig, h, iter)$x
}
out_time <- Sys.time()
out_time - in_time

acc_rate_diffusive <- mean(RWMMixture1d(x0[1], p, mu, sig, h, 1e7L)$acc)
acc_rate_diffusive

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
}

df <- data.frame(cbind(quants, iters, probs))
names(df) <- c("quantile", "iteration", "probability")

diffusive <- ggplot(data = df, aes(x = iteration, y = quantile, color = probability, group = probability)) +
  geom_line(size = 0.3) +
  xlab("Iteration") +
  ylab("Quantile") +
  theme_bw() + 
  theme(legend.position = "none")
diffusive 

ggsave(filename = "rwm_mixture_smallstep.pdf",
       plot = diffusive,
       width = 8, height = 8, units = "cm")

#####
# Nondiffusive RWM, large step size
#####

# MCMC parameters ##
iter <- 30L    # Iteration count 
R    <- 1e6L   # Replicate count
h <- 6      # Step size

x0 <- 2 * DrawFromGaussianMixture(R, p, mu, sig)
out <- matrix(NA, nrow = R, ncol = iter + 1)

in_time <- Sys.time()

for(i in 1:R) {
  out[i, ] <-  RWMMixture1d(x0[i], p, mu, sig, h, iter)$x
}
out_time <- Sys.time()
out_time - in_time

acc_rate_nondiffusive <- mean(RWMMixture1d(x0[1], p, mu, sig, h, 1e7L)$acc)
acc_rate_nondiffusive

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
}

df <- data.frame(cbind(quants, iters, probs))
names(df) <- c("quantile", "iteration", "probability")

nondiffusive <- ggplot(data = df, aes(x = iteration, y = quantile, color = probability, group = probability)) +
  geom_line(size = 0.3) +
  xlab("Iteration") +
  ylab("Quantile") +
  theme_bw() + 
  theme(legend.position = "none")
nondiffusive

ggsave(filename = "rwm_mixture_largestep.pdf",
       plot = nondiffusive,
       width = 8, height = 8, units = "cm")



#####
# Skewed start, small step size
#####

R <- 1e6L
iter <- 100L
h <- 2
x0 <- 5 + DrawFromGaussianMixture(R, p, mu, sig)
out <- matrix(NA, nrow = R, ncol = iter + 1)

in_time <- Sys.time()

for(i in 1:R) {
  out[i, ] <-  RWMMixture1d(x0[i], p, mu, sig, h, iter)$x
}
out_time <- Sys.time()
out_time - in_time

thin <- 1L

prob <- c(seq(0.05, 0.45, length.out = 10), seq(0.55, 0.95, length.out = 10))
iters_plot <- seq.int(0, iter, thin)

num_prob <- length(prob)
iters <- rep(iters_plot, each = num_prob)
probs <- rep(prob, length(iters_plot))
quants <- NULL
for(i in iters_plot) {
  quants <- c(quants, quantile(out[, i + 1], probs = prob))
}

df <- data.frame(cbind(quants, iters, probs))
names(df) <- c("quantile", "iteration", "probability")

skewed <- ggplot(data = df, aes(x = iteration, y = quantile, color = probability, group = probability)) +
  geom_line(size = 0.3) +
  xlab("Iteration") +
  ylab("Quantile") +
  theme_bw() + 
  theme(legend.position = "none")
skewed

ggsave(filename = "rwm_mixture_smallskewed.pdf",
       plot = skewed,
       width = 8, height = 8, units = "cm")

