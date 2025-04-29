#############################################################
# Benchmark for empirical Wasserstein distance computations #
#############################################################
# We'll measure the computing costs of:
#   1) The cost matrix.
#   2) Various assignment solvers.
#   3) Computing the jackknife assignment costs.

library(wassersteinbound)
library(foreach)
library(doParallel)

###
# 0. Experiment setup
###

# Hard coded key-value pairs for output and plot. Do NOT change these.
computation <- c(
  "Cost matrix",
  "Assignment (TOMS 1015)",
  "Assignment (Simplex)",
  "Jackknife"
)
cost_mat_key <- 1
solvers <- c("Assignment (TOMS 1015)", "Assignment (Simplex)")
solver_keys <- c(2, 3)
jackknife_key <- 4

time_assignment <- function(solver,x,y) {
  if(solver == "Assignment (Simplex)") {
    out <- assignment_squared_euclidean_networkflow_cpp(x,y,FALSE)$timing_nanoseconds * 1e-9 # Don't compute the assignment
  } else if(solver == "Assignment (TOMS 1015)") {
    out <- assignment_squared_euclidean_toms1015_cpp(x,y,TRUE)$timing_nanoseconds * 1e-9 # Use the (very fast) epsilon-scaling heuristic
  } else if(solver == "Assignment (LAPJV)") {
    out <- assignment_squared_euclidean_toms1015_cpp(x,y,FALSE)$timing_nanoseconds * 1e-9   # Usual Jonker-Volgenant without the pre-solve heuristic == TOMS1015 without epsilon-scaling
  }
  return(list("seconds" = diff(out)))
}

time_jackknife <- function(x,y) {
  out <- assignment_squared_euclidean_jackknife_cpp(x,y)$timing_nanoseconds * 1e-9
  return(list("seconds" = diff(out)))
}

format_df <- function(df_){
  df_ <- data.frame(df_)
  names(df_) <- c("d", "n", "Computation", "walltime")
  df_$Computation <- computation[df_$Computation]
  df_$d <- factor(df_$d)
  df_
}


# Parallel computing
ncores <- 32
cl <- parallel::makeCluster(ncores)
registerDoParallel(cl)

# RNG seed
seed <- 19032024

# Replicate count, dimensions, and sample size
R <- 8
d <- rev(c(2, 8, 32, 128, 512))
n <- rev(c(128, 256, 512, 1024, 2048, 4096, 8192, 16384))

# Empirical measures are drawn from \otimes^d N(0, \sigma_x^2) and \otimes^d N(0, \sigma_y^2)
sigma_x <- 1
sigma_y <- 2

# We need to manually handle seeds for nested for-loops
rng <- rngtools::RNGseq(R, seed)
# Handle the R = 1 case if needed
if(R == 1) rng <- list(rng)

###
# 1. Assignment problem and cost matrix
###
timing_assignment <- 
foreach(j = 1:length(n), .combine = "rbind", .packages = "wassersteinbound") %:% 
  foreach(i = 1:length(d), .combine = "rbind") %:%
  foreach(solver_key = solver_keys, .combine = "rbind") %:%
  foreach(k = 1:R, r = rng, .combine = "rbind") %dopar% {
    rngtools::setRNG(r)
    
    x <- matrix(rnorm(d[i] * n[j], sd = sigma_x), nrow = n[j]) # NB: non-negligible chunk of the computing cost for large n,d
    y <- matrix(rnorm(d[i] * n[j], sd = sigma_y), nrow = n[j])
    
    seconds <- time_assignment(computation[solver_key], x, y)$seconds
    rbind(c(d[i], n[j], solver_key, seconds["assignment"]),
          c(d[i], n[j], cost_mat_key, seconds["cost_matrix"]))
  }
timing_assignment <- format_df(timing_assignment)

###
# 2. Jackknife assignment costs
###
n_jack <- rev(c(128, 256, 512, 1024, 2048, 4096)) # Cubic scaling, so skip the larger sample sizes

timing_jackknife <- 
  foreach(j = 1:length(n_jack), .combine = "rbind", .packages = "wassersteinbound")  %:% 
  foreach(i = 1:length(d), .combine = "rbind") %:% 
  foreach(k = 1:R, r = rng, .combine = "rbind") %dopar% { 
    rngtools::setRNG(r)
    
    x <- matrix(rnorm(d[i] * n_jack[j], sd = sigma_x), nrow = n_jack[j])
    y <- matrix(rnorm(d[i] * n_jack[j], sd = sigma_y), nrow = n_jack[j])
    
    seconds <- time_jackknife(x, y)$seconds
    
    c(d[i], n_jack[j], jackknife_key, seconds["jackknife"])
  }
timing_jackknife <- format_df(timing_jackknife)

parallel::stopCluster(cl)

save(R, timing_assignment, timing_jackknife, file = "benchmark.RData")
