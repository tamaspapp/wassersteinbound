
library(wassersteinbound)
library(doParallel)
library(rngtools)
library(reshape2)
library(ggplot2)
library(ggh4x)
library(latex2exp)

reps <- 100
D <- 10^c(1,2)
N <- 10^c(1,2,3)
Sigma_sq <- c(1.1, 2, 10)

seed <- 12345
set.seed(seed)
rng <- RNGseq(reps, seed)

SquareSign <- function(x) { return(sign(x) * x^2)}
GetExactW2sq <- function(d, sigma_sq) {return(d * (sqrt(sigma_sq) - 1) ^ 2)}

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
  
  x1 <- matrix(rnorm(n * d, sd = sqrt(sigma_sq)), nrow = n, ncol = d)
  y  <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y1 <- matrix(rnorm(n * d), nrow = n, ncol = d)
  
  w2sq_x1y <- w2sq_empirical(x1, y)$w2sq
  w2sq_y1y <- w2sq_empirical(y1, y)$w2sq
  
  empirical <- w2sq_x1y
  U <- w2sq_x1y - w2sq_y1y
  L <- SquareSign(sqrt(w2sq_x1y) - sqrt(w2sq_y1y))
  
  return(data.frame("d" = d, "n" = n, "sigma_sq" = sigma_sq, 
                    "U" = U, "L" = L, "Plugin" = empirical, "Exact" = GetExactW2sq(d, sigma_sq)))
}
parallel::stopCluster(cl)

estim <- melt(out, id.vars = c("d", "n", "sigma_sq", "Exact"), 
            measure.vars = c("U", "L", "Plugin"),
            value.name = "w2sq", variable.name = "Estimator")
save(estim, out, file = "prelim.Rdata")

# Plot ####
load("prelim.Rdata")

# Reorder the estimators
estim$Estimator <- factor(estim$Estimator , levels=c("Plugin", "U", "L"))
levels(estim$Estimator)[1] <- "Plug-in"


# Label the different facets
sigma_sq_label_fun <- function(string) TeX(paste0("$\\sigma^2 = $",string))
sigma_sq_label <- as_labeller(sigma_sq_label_fun, default = label_parsed)
d_label_fun <- function(string) TeX(paste0("$d = $",string))
d_label <- as_labeller(d_label_fun, default = label_parsed)

plt <-
ggplot(estim, aes(x = factor(n), y = w2sq, color = Estimator, group = interaction(n, Estimator))) +
  ggh4x::facet_grid2(d~sigma_sq, scales = "free", independent = "y",
                     labeller = labeller(sigma_sq = sigma_sq_label, d = d_label)) +
  geom_hline(aes(yintercept = Exact, linetype = "Exact squared Wasserstein")) +
  geom_boxplot(width = 0.75) +
  labs(y = "Squared Wasserstein distance", x = "n") +
  scale_linetype_manual(values = "solid", breaks = "Exact squared Wasserstein") +
  scale_y_continuous(minor_breaks = NULL) +
  theme_bw() +
  guides(linetype=guide_legend(title=element_blank(), order=2),
         color=guide_legend(order=1))
plt

ggsave(filename = "prelim_comparison.pdf",
       plot = plt,
       width = 24, height = 8, units = "cm")
