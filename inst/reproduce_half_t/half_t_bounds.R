library(wassersteinbound)
library(dplyr)

source('../bias_helpers.R') # Auxiliary helpers

# Read data from disk ####

# File is set up for command line usage
args <- commandArgs(trailingOnly = TRUE)
epsilon <- as.double(args[1])

load(file = paste0("half_t_epsilon=",epsilon,".Rdata"))
load(file = paste0("half_t_singlechain_exact.Rdata"))
#####

# Set up empirical bounds ####
extra_thinning <- 10
aggregate <- function(x) aggregate_mcmc_samples(x, extra_thinning)

thinning_factor <- thin * extra_thinning
cat(paste0("Effective thinning factor for proposed estimators: ", thinning_factor,".",
           "\n","[IACT in worst coordinate is roughly 125 based on preliminary runs.]"))
cat(paste0("Sample size for proposed estimators: ",no_chains*(1 + (chain_length-1)%/%thinning_factor),".\n"))
#####

# Get estimators ####
# Empirical estimators
empirical <- get_empirical_bound_nohedge(aggregate(coupled_output$ys), aggregate(coupled_output$xs), 
                                         aggregate(singlechain_output$ys), num_blocks = no_chains)

get_var_decrease <- function(df) {
  df %>% group_by(Estimator) %>% mutate(var_improvement = (stderr_nocoupling / stderr)^2, .keep = "none")
}
var_improvement <- get_var_decrease(empirical)

# Coupling bound
cub <- get_cub(coupled_output)

# Lower bound (only product bound, since Gelbrich bound is looser)
lb <- product_lb(coupled_output, jackknife = T)
lb <- data.frame("Estimator" = "Tractable lower bound", 
                 "w2sq" = lb$estimate, 
                 "stderr" = lb$jack_stderr, 
                 "bias" = lb$jack_bias)

# Collate estimators, go from point estimates + standard errors to confidence intervals
estimators <- dplyr::bind_rows(empirical, cub, lb)

# Trace of covariance as measure of overall scale
trace_of_cov <- mean(sapply(seq_along(coupled_output$xs), function(i) sum(apply(rbind(coupled_output$ys[[i]], singlechain_output$ys[[i]]),2,var))))
#####

# Save ####
save(epsilon, estimators, var_improvement, trace_of_cov, file = paste0("half_t_epsilon=",epsilon,"_estimators.Rdata"))
#####