library(wassersteinbound)
library(doParallel)
library(purrr)

source('half_t_functions.R')

# Import Riboflavin data ####
load("../datasets/riboflavin.RData_")

X <- as.matrix(riboflavin$x); colnames(X) <- rownames(X) <- NULL
X <- scale(X, center = T, scale = F) # Center covariates, no scaling so as to leave prior unaffected

X_transpose <- t(X)

y <- riboflavin$y
y <- scale(y, center = T, scale = F) # Center response, no scaling so as to leave prior unaffected

n <- dim(X)[1]
p <- dim(X)[2]
#####

# Simulation parameters ####
# Half-t prior: degrees of freedom
t_dist_df <- 2 

chain_length <- 10000
burnin <- 1000
#####

# See how the posteriors for xi look like with different values of approximation parameter ####

epsilons <- c(0, 0.0003, 0.001, 0.003, 0.01, 0.03)

cl <- parallel::makeCluster(6); registerDoParallel(cl)
# "dopar" because we use the same seed in each chain
xis <- foreach(eps=epsilons, .combine = "rbind", .packages = "wassersteinbound") %dopar% {
  set.seed(1)
  xi <- half_t_mcmc(chain_length + burnin, burnin, X, t(X), y, t_dist_df=t_dist_df, approximate_algo_delta=eps)$xi_samples
  data.frame("xi" = xi, "epsilon" = eps)
}
stopCluster(cl)

save(xis, file="half_t_xi_overdisp.RData")

# Plotting ####
library(ggplot2)
library(latex2exp)

scientific_but_for_zero <- function(x){
  x <- as.numeric(as.character(x))
  ifelse(x != 0, format(x, scientific=T), x)
}

load(file="half_t_xi_overdisp.RData")
xis$epsilon <- factor(xis$epsilon)

plt <-
  ggplot(xis, aes(x=xi, color = epsilon, group = epsilon)) +
  stat_density(geom="line",position="identity") +
  scale_x_log10() +
  scale_color_manual(values = tail(RColorBrewer::brewer.pal(7,"YlOrRd"),6),
                     labels = scientific_but_for_zero) +
  theme_bw(base_size=12) +
  labs(x = TeX("State $\\xi$"), y = "Density", color = TeX("Parameter $\\epsilon$")) +
  theme(panel.grid.minor = element_blank())
plt

ggsave(filename = "overdisp_xi_half_t.pdf",
       plot = plt,
       width = 20, height = 9, units = "cm")
