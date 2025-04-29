library(wassersteinbound)
library(doParallel)
library(rngtools)

reps <- 100
D <- rev(c(10, 100))
N <- rev(c(10, 100, 1000))
Sigma_sq <- seq(1, 4, length = 7)

seed <- 12345
set.seed(seed)
rng <- RNGseq(reps, seed)

GetExactW2sq <- function(d, sigma_sq) return(d/2 * ( (sqrt(sigma_sq) - 1)^2 + (sqrt(sigma_sq) - 2)^2 ))

ncores <- 4
cl <- parallel::makeCluster(ncores)
doParallel::registerDoParallel(cl)

out <-
  foreach(d = D, .combine = "rbind") %:%
  foreach(n = N, .combine = "rbind") %:%
  foreach(sigma_sq = Sigma_sq, .combine = "rbind") %:%
  foreach(i = 1:reps, rng_ = rng, .combine = "rbind", .packages = c("wassersteinbound")) %dopar% {
    # Set RNG
    rngtools::setRNG(rng_)
    
    x <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = n, ncol = d)
    x1 <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = n, ncol = d)
    y <- matrix(rnorm(n * d, sd = c(1,2)), nrow = n, ncol = d, byrow = T)
    y1 <- matrix(rnorm(n * d, sd = c(1,2)), nrow = n, ncol = d, byrow = T)
    
    w2sq_x1y <- w2sq_empirical(x1, y)$w2sq
    w2sq_y1x <- w2sq_empirical(y1, x)$w2sq
    
    w2sq_y1y <- w2sq_empirical(y1, y)$w2sq
    w2sq_x1x <- w2sq_empirical(x1, x)$w2sq
    
    U <- w2sq_x1y - w2sq_y1y
    V <- pmax(w2sq_x1y - w2sq_x1x, w2sq_y1x - w2sq_y1y)
    
    return(data.frame("d" = d, "n" = n, "sigma_sq" = sigma_sq, 
                      "U" = U, "V" = V, "Exact" = GetExactW2sq(d, sigma_sq)))
  }
parallel::stopCluster(cl)

save(out, D, file = "prelim_overdisp.Rdata")


# Plot #### 
load("prelim_overdisp.Rdata")

library(reshape2)
library(ggplot2)
library(ggh4x)
library(latex2exp)

estim <- melt(out, id.vars = c("d", "n", "sigma_sq", "Exact"), 
              measure.vars = c("U", "V"),
              value.name = "w2sq", variable.name = "Estimator")

# Set negative values to zero
estim$w2sq[estim$w2sq < 0] <- 0


# Nicer labels for facets facets
  # Estimators
estim_labels <- c(
  U = TeX("Estimator $U(\\bar{\u{03BC}}_n, \u{03BC}_n, \u{03BD}_n)$"),
  V = TeX("Estimator $V(\u{03BC}_n, \u{03BD}_n, \\bar{\u{03BC}}_n, \\bar{\u{03BD}}_n)$")
)
levels(estim$Estimator) <- estim_labels
  # Dimension
d_labeller <- function(string) TeX(paste0("$d = $",string))
d_labels <- d_labeller(sort(D))
names(d_labels) <- sort(D)

estim$d <- as.factor(estim$d)
levels(estim$d) <- d_labels

  
plt <-
  ggplot(estim, aes(x = factor(sigma_sq), y = w2sq/Exact, color = factor(n), group = interaction(sigma_sq, n))) +
  geom_hline(yintercept = 1) +
  geom_boxplot(width = 0.75) +
  facet_grid(d~Estimator, scales = "free_y", labeller = label_parsed) +
  scale_y_continuous(minor_breaks = NULL) +
  labs(y = "Size relative to squared Wasserstein distance", x = TeX("$\\sigma^2$"), color = "n") +
  theme_bw()
plt

ggsave(filename = "prelim_overdisp.pdf",
       plot = plt,
       device = cairo_pdf,
       width = 24, height = 10, units = "cm")
