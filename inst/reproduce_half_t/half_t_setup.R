library(wassersteinbound)
library(doRNG)
library(purrr)

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

no_chains <- 100
burnin <- 1000 # 1000 burn-in iterations suffice, by Figure 5.5(c) in https://arxiv.org/abs/2206.05691v3
chain_length <- 5000 # Chain length kept, starting from t = "burnin".
thin <- 5

# Randomness
seed <- 12345