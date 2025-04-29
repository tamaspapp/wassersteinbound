library(wassersteinbound)
library(dplyr)
library(doParallel)

#####
# Miscellaneous helpers
#####
average_over_list <- function(lst) Reduce("+", lst) / length(lst)

sample_gaussian_rows <- function(n, gauss_params) {
  d <- dim(gauss_params$Sigma)[2]
  samples <- matrix(rnorm(n * d), ncol = d) %*% chol(gauss_params$Sigma) # N(0, Sigma) rows
  samples <- sweep(samples, 2, gauss_params$mu, "+") # Add the mean to all rows
  return(samples)
}

aggregate_mcmc_samples <- function(xs, thinning_factor, samples_per_chain = NA) {
  no_chains <- length(xs)
  no_iterations <- nrow(xs[[1]])
  d <- ncol(xs[[1]])
  
  # Determine number of samples that are kept within each chain
  if(is.na(samples_per_chain)) samples_per_chain <- 1 + floor((no_iterations-1)/thinning_factor)
  
  # Indices of samples that are kept within each chain
  select_within_chain <- 1 + thinning_factor * seq.int(0, samples_per_chain-1)
  
  sample_size <- no_chains * samples_per_chain
  
  # Thin samples and concatenate them
  x <- matrix(NA, nrow = sample_size, ncol = d)
  for(r in 1:no_chains) x[(r-1) * samples_per_chain + 1:samples_per_chain, ] <- xs[[r]][select_within_chain, ]
  return(x)
}

stderr <- function(x) sqrt(var(x)/length(x))
sgnsq <- function(x) sign(x) * x^2

#####
# Get bounds from MCMC output
#####

# Get confidence intervals, also convert L-bar to L
convert_and_get_ci <- function(df, z = qnorm(0.975)){
  # Correct for bias
  ifelse(df$Estimator == "Tractable lower bound", sgnsq(df$w2), df$w2sq)
  data.frame("w2sq" = ifelse(df$Estimator == "Lbar", sgnsq(df$w2), df$w2sq),
             "ci_up" = ifelse(df$Estimator == "Lbar", sgnsq(df$w2 + z * df$stderr), df$w2sq + z * df$stderr),
             "ci_lo" = ifelse(df$Estimator == "Lbar", sgnsq(df$w2 - z * df$stderr), df$w2sq - z * df$stderr),
             "Estimator" = ifelse(df$Estimator == "Lbar", "L", df$Estimator))
}

# Coupling upper bound
get_cub <- function(coupled_output){
  cubs <- unlist(lapply(coupled_output$squaredist, mean))
  return(data.frame("w2sq" = mean(cubs), 
                    "stderr" = sd(cubs) / sqrt(length(cubs)), 
                    "Estimator" = "Coupling"))
}

# Plug-in estimators
get_u <- function(plugin, debias, normalized_var = function(x) var(x) / length(x)) {
  pot.plugin <- plugin$potentials_x + plugin$potentials_y
  pot.debias <- debias$potentials_x + debias$potentials_y
  data.frame("w2sq" = plugin$w2sq - debias$w2sq,
             "stderr" = sqrt(normalized_var(pot.plugin - pot.debias)),
             "stderr_nocoupling" = sqrt(
               normalized_var(plugin$potentials_x) + 
                 normalized_var(debias$potentials_x) + 
                 normalized_var(plugin$potentials_y - debias$potentials_y)
             ),
             "Estimator" = "U")
}
get_lbar <- function(plugin, debias, normalized_var = function(x) var(x) / length(x)) {
  pot.plugin <- plugin$potentials_x + plugin$potentials_y
  pot.debias <- debias$potentials_x + debias$potentials_y
  data.frame("w2sq" = sqrt(plugin$w2sq) - sqrt(debias$w2sq),
             "stderr" = sqrt(normalized_var(0.5*pot.plugin/sqrt(plugin$w2sq) - 0.5*pot.debias/sqrt(debias$w2sq))),
             "stderr_nocoupling" = sqrt(
               normalized_var(0.5*plugin$potentials_x/sqrt(plugin$w2sq)) + 
                 normalized_var(0.5*debias$potentials_x/sqrt(debias$w2sq)) + 
                 normalized_var(0.5*plugin$potentials_y/sqrt(plugin$w2sq) - 0.5*debias$potentials_y/sqrt(debias$w2sq))
             ),
             "Estimator" = "Lbar")
}



# We use the estimator W here.
#   Ensure that x and y are correlated to get a reduction in variance
get_empirical_bound <- function(x, y, x_prime, y_prime, num_blocks){
  # x = n x d matrix; 
  #     rows are several contiguous blocks of "num_blocks" samples per chain
  
  # Accounting for MCMC autocorrelation in standard errors
  n <- dim(x)[1]
  if(n %% num_blocks != 0) stop("Number of blocks must divide sample size evenly.")
  iters_per_chain <- n/num_blocks
  stride <- seq(0,n-1,by=iters_per_chain)
  blockwise_mean <- function(x) sapply(stride, function(i) mean(x[i + 1:iters_per_chain]))
  normalized_var <- function(x){y <- blockwise_mean(x); var(y)/length(y)}
  
  # Squared distances
  w2sq_x_yprime <- w2sq_empirical(x, y_prime)
  w2sq_y_yprime <- w2sq_empirical(y, y_prime)
  
  w2sq_y_xprime <- w2sq_empirical(y, x_prime)
  w2sq_x_xprime <- w2sq_empirical(x, x_prime)
  
  # Estimators (with hedging) and standard errors
  if(w2sq_x_yprime$w2sq - w2sq_y_yprime$w2sq > 
     w2sq_y_xprime$w2sq - w2sq_x_xprime$w2sq) { 
    plugin <- w2sq_x_yprime
    debias <- w2sq_y_yprime # Center with y
  } else {
    plugin <- w2sq_y_xprime
    debias <- w2sq_x_xprime # Center with x
  }
  
  u <- get_u(plugin, debias, normalized_var); u$Estimator <- "V" # Since we've hedged
  lbar <- get_lbar(plugin, debias, normalized_var)
  return(dplyr::bind_rows(u,lbar))
}

# Estimator U.
get_empirical_bound_nohedge <- function(x, y, x_prime, num_blocks){
  # x = n x d matrix; 
  #     rows are several contiguous blocks of "num_blocks" samples per chain
  
  # Accounting for MCMC autocorrelation in standard errors
  n <- dim(x)[1]
  if(n %% num_blocks != 0) stop("Number of blocks must divide sample size evenly.")
  iters_per_chain <- n/num_blocks
  stride <- seq(0,n-1,by=iters_per_chain)
  blockwise_mean <- function(x) sapply(stride, function(i) mean(x[i + 1:iters_per_chain]))
  normalized_var <- function(x){y <- blockwise_mean(x); var(y)/length(y)}
  
  # Squared distances
  w2sq_y_xprime <- w2sq_empirical(y, x_prime)
  w2sq_x_xprime <- w2sq_empirical(x, x_prime)
  
  # Estimators and standard errors
  u <- get_u(w2sq_y_xprime, w2sq_x_xprime, normalized_var)
  lbar <- get_lbar(w2sq_y_xprime, w2sq_x_xprime, normalized_var)
  
  return(dplyr::bind_rows(u,lbar))
}

# Lower bounds that exploit structure ####
get_lowerbound <- function(coupled_out, jackknife = T, convergence_rate = 1){
  gelbrich_lb <- gelbrich_lb(coupled_out, jackknife, convergence_rate)
  product_lb <- product_lb(coupled_out, jackknife, convergence_rate)
  
  # Output the largest bound
  if(jackknife){
    # Use bias-corrected estimate
    if(gelbrich_lb$estimate - gelbrich_lb$jack_bias > product_lb$estimate - product_lb$jack_bias)
      out <- gelbrich_lb
    else out <- product_lb
  } else {
    # Don't have bias estimates
    if(gelbrich_lb$estimate > product_lb$estimate)
      out <- gelbrich_lb
    else out <- product_lb
  }
  
  return(data.frame("w2sq" = out$estimate, "stderr" = out$jack_stderr, "bias" = out$jack_bias,
                    "Estimator" = "Tractable lower bound"))
}

# Gelbrich lower bound
gelbrich_lb <- function(coupled_output, jackknife = T, convergence_rate=1){#, ncores = 1){
  xs <- coupled_output$xs
  ys <- coupled_output$ys
  
  n <- length(xs)
  
  if(!jackknife | n == 1){
    approx_mean <- average_over_list(lapply(xs, colMeans)) 
    approx_cov <- average_over_list(lapply(xs, cpp_cov))

    exact_mean <- average_over_list(lapply(ys, colMeans)) 
    exact_cov <- average_over_list(lapply(ys, cpp_cov))

    # Gelbrich lower bound
    w2sq_gelbrich <- w2sq_gaussian_cpp(exact_mean, exact_cov, approx_mean, approx_cov)
    w2sq_gelbrich_jack_bias <- NA
    w2sq_gelbrich_jack_stderr <- NA
  } else {
    # Approximation
    approx_means <- lapply(xs, colMeans)
    approx_covs <- lapply(xs, cpp_cov)
    
    approx_mean <- average_over_list(approx_means) 
    approx_cov <- average_over_list(approx_covs)
    
    # Exact
    exact_means <- lapply(ys, colMeans)
    exact_covs <- lapply(ys, cpp_cov)
    
    exact_mean <- average_over_list(exact_means)
    exact_cov <- average_over_list(exact_covs)
    
    # Gelbrich lower bound + jackknife stderr and bias estimates
    w2sq_gelbrich <- w2sq_gaussian_cpp(exact_mean, exact_cov, approx_mean, approx_cov)
    
    # Jackknife bias and standard error
    # O(nd^3) complexity (could be parallelized)
    loo <- function(i){ 
        # Compute LOO summary statistics; complexity O(d^2).
        mu1    <- (exact_mean * n - exact_means[[i]]) / (n-1) 
        Sigma1 <- (exact_cov * n - exact_covs[[i]]) / (n-1)
        mu2    <- (approx_mean * n - approx_means[[i]]) / (n-1)
        Sigma2 <- (approx_cov * n - approx_covs[[i]]) / (n-1)
        return(w2sq_gaussian_cpp(mu1, Sigma1, mu2, Sigma2)) # Complexity O(d^3).
    }
    loo_estimates <- sapply(1:n, loo)
    
    w2sq_gelbrich_jack_bias <- jack_bias(loo_estimates, w2sq_gelbrich, convergence_rate)
    w2sq_gelbrich_jack_stderr <- jack_stderr(loo_estimates, w2sq_gelbrich)
  }
  
  return(list("estimate" = w2sq_gelbrich, 
              "jack_stderr" = w2sq_gelbrich_jack_stderr, 
              "jack_bias" = w2sq_gelbrich_jack_bias,
              "x_mean" = approx_mean,
              "y_mean" = exact_mean,
              "x_cov" = approx_cov,
              "y_cov" = exact_cov))
}

product_lb <- function(coupled_output, jackknife = T, convergence_rate=1){
  
  xs <- coupled_output$xs
  ys <- coupled_output$ys
  
  n <- length(xs)         # Number of chains
  iter <- dim(xs[[1]])[1] # Number of samples within each chain 

  # Bind to matrix
  x_mat <- do.call(rbind, xs)
  y_mat <- do.call(rbind, ys)

  if(!jackknife | n == 1){
    # Sort the samples in each column==dimension
    x_mat <- apply(x_mat,2,sort)
    y_mat <- apply(y_mat,2,sort)
    
    # Calculate the estimator
    w2sq_prod <- sum((x_mat - y_mat)^2) / (n * iter)
    jack_bias <- NA
    jack_stderr <- NA
  } else {
    # Optimized leave-one-out calculation:
    #    - Suppose that we have sorted the samples in each column (i.e. dimension).
    #      To remove a sample from the i-th chain, we need to keep track of its
    #      rank, which corresponds to its index in the sorted vector of samples.
    #
    #      So, if we sort once and we keep track of the ranks, then we can get
    #      the jackknife estimators cheaper, as all other operations are additions.
    #      
    #    - Implementation: 
    #         (1) rank colwise
    #         (2) in-place sort based on rank
    #         (3) efficient summation based on indices of removed samples
  
    # (1) Get ranks of the samples  ||  0-index for C compatibility
    ranks_x <- colwise_rank_cpp(x_mat) # apply(x_mat,2,function(x)rank(x, ties.method = "first"))-1L 
    ranks_y <- colwise_rank_cpp(y_mat) # apply(y_mat,2,function(x)rank(x, ties.method = "first"))-1L
    mode(ranks_x) <- mode(ranks_y) <- "integer"
    
    # (2) Sort samples in-place based on ranks
    permute_matrix_colwise_inplace_cpp(x_mat, ranks_x) 
    permute_matrix_colwise_inplace_cpp(y_mat, ranks_y)
    
    # Get estimator
    w2sq_prod <- sum((x_mat - y_mat)^2) / (n * iter)
    
    # Get jackknife estimates
    loo_w2sq <- function(i) {
      remove <- 1:iter + (i-1)*iter
      # (3) Efficient summation based on indices of removed samples
      #     Cpp function equivalent to: apply(1:cols, function(col) sum((x_mat[-ranks_x[remove,col], col] - y_mat[-ranks_y[remove,col], col])^2)) 
      sum(colwise_squared_norm_with_skip_cpp(x_mat, y_mat, ranks_x[remove, ], ranks_y[remove, ]))/((n-1) * iter)
    }
    loo_estimates <- sapply(1:n, loo_w2sq)
    
    jack_bias <- jack_bias(loo_estimates, w2sq_prod, convergence_rate)
    jack_stderr <- jack_stderr(loo_estimates, w2sq_prod)
  }

  return(list("estimate" = w2sq_prod,
              "jack_stderr" = jack_stderr,
              "jack_bias" = jack_bias))
}

# Jackknife standard error
jack_stderr <- function(loo_estimates, estimate){
  n <- length(loo_estimates)
  sqrt((n-1)^2/n) * sd(loo_estimates)
}
# Jackknife bias
jack_bias <- function(loo_estimates, estimate, rate = 1){
  n <- length(loo_estimates)
  
  ## Usual jackknife bias estimate assumes a "parametric" rate (1/n).
  ##
  ## A different correction is needed when the rate of convergence is slower,
  ## see Section 2.2 of https://doi.org/10.2307/2334280.
  ##
  ## If rate of convergence is n^(-rate), use:
  ## (n-1)^rate * (mean(loo_estimates) - estimate) / (n^rate - (n-1)^rate)
  ##
  ## To first order, this is:
  ((n-1)/rate) * (mean(loo_estimates) - estimate) # We recover the usual correction for rate = 1.
}


