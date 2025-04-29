get_problem_params <- function(ds, hs){
  foreach(d = ds, h = hs) %do% {
    
    # Get exact target parameters
    mu <- rep(0, d)
    target_params <- AR1_target_params(d, rho)
    
    Sigma        <- target_params$Sigma
    Sigma_chol_u <- chol(Sigma)
    Omega        <- target_params$Omega
    Omega_chol_u <- target_params$Omega_chol_u
    
    # Get ULA target parameters
    Sigma_ULA <- as(solve((diag(d) - h^2 / 4 * Omega) %*% Omega), "matrix")
    Sigma_ULA_chol_u <- chol(Sigma_ULA)
    
    # Get initial distribution parameters
    mu_0 <- mu
    Sigma_0 <- diag(rep(3, d))
    Sigma_0_chol_u <- chol(Sigma_0)
    
    # Get friction parameters
    gammas <- c("Overdamped" = Inf, 
                "Underdamped" = 2 / max(eigen(Sigma,T,T)$values)) # Critical damping
    
    return(list("target_type"  = "sparse_gaussian",
                "mu"           = mu,
                "Sigma"        = Sigma,
                "Omega"        = Omega,
                "Omega_chol_u" = Omega_chol_u,
                "Sigma_chol_u" = Sigma_chol_u,
                "mu_0"         = mu_0,
                "Sigma_0"      = Sigma_0,
                "Sigma_0_chol_u" = Sigma_0_chol_u,
                "h"                = h,
                "d"                = d,
                "gammas"           = gammas,
                "Sigma_ULA"        = Sigma_ULA, 
                "Sigma_ULA_chol_u" = Sigma_ULA_chol_u))
  }
}
get_empirical_w2sq_upper <- function(empirical_out, debias_start_iter, debias_end_iter){
  empirical_out %>%
    dplyr::group_by(gamma, d) %>%
    dplyr::mutate(w2sq = w2sq - mean(w2sq[(iter >= debias_start_iter) & (iter <= debias_end_iter)]))
}
get_empirical_w2sq_lower <- function(empirical_out, debias_start_iter, debias_end_iter){
  
  signed_square <- function(x) sign(x) * x^2
  
  empirical_out %>%
    dplyr::group_by(gamma, d) %>%
    dplyr::mutate( w2sq = signed_square( sqrt(w2sq) - sqrt(mean(w2sq[(iter >= debias_start_iter) & (iter <= debias_end_iter)])) ) )
}
get_mixing_times <- function(output, w2sq_thresh){
  output %>%
    dplyr::group_by(gamma, d) %>%
    dplyr::filter(w2sq < w2sq_thresh) %>%
    dplyr::slice(which.min(iter))
}


# 
# library(Matrix)
# d <- 10
# 
# d <- params$d
# Omega <- params$Omega
# 
# O <- bdiag(Diagonal(d), eta * Diagonal(d)) # partial momentum refresh
# 
# B <- bdiag(Diagonal(d), Diagonal(d))
# B[1:d, (d+1):(2*d)] <- (-0.5) * delta * Omega  # momentum update
# 
# A <- bdiag(Diagonal(d), Diagonal(d))
# A[1:d, 1:d] <- delta # position update
# 
# slope_mat <- as.matrix(B %*% A %*% B %*% O)
# 
# 
# ############
# # Sanity check: stationary distribution factorizes over position and momentum 
# 
# ##############





#####
# Exact squared Wasserstein convergence of OBAB discretization
#####
#  -Target and initial distribution are Gaussian and simultaneously diagonalizable.
# - For the considered discretization, this means that all marginals are also
# simultaneously diagonalizable.
#
# - We therefore treat the marginals separately. Each "marginal" is a pair ("x", "v")
# consisting of a 1-dimensional position "x" and a 1-dimensional velocity "v".
#####

library(expm) # For (integer) matrix powers in exact squared Wasserstein distance calculation

# Starting and target covariances (Sigma_0 and Sigma_target) must be must be simultaneously diagonalizable.
get_obab_exact_w2sq <- function(iter, thin,
                                delta, gamma,
                                mu_0, Sigma_0_diag,         # Sigma_0_diag      = "diagonal" of starting covariance Sigma_0.
                                mu_infty, Sigma_target_diag # Sigma_target_diag = "diagonal" of the TARGET covariance Sigma_target (NOT the stationary covariance).
                                ){
  # Get slope matrix for the recursion
  obab_slope_matrix <- function(sig_sq) {
    #   sig_sq = target variance
    #   delta = step size in obab discretization
    #   gamma = friction parameter

    # Partial velocity refreshment: diagonal matrix
    eta <- exp(-delta * gamma)
    P <- diag(c(1, eta))

    # Leapfrog dynamics: rotation matrix
    omega_sq <- 1/ sig_sq

    R <- matrix(nrow = 2, ncol = 2)
    R[1,1] <- 1 - (delta^2 / 2) * omega_sq
    R[1,2] <- delta
    R[2,1] <- - delta * omega_sq * (1 - (delta^2 / 4) * omega_sq)
    R[2,2] <- 1 - (delta^2 / 2) * omega_sq

    # Compose the dynamics:
    B <- R %*% P # This (correctly) corresponds to partial refreshment being the first step.
    return(B)
  }

  # # Get slope matrix for the recursion
  # obab_slope_and_intercept <- function(sig_sq, delta, gamma) {
  #   #   sig_sq = target variance
  #   #   delta = step size in obab discretization
  #   #   gamma = friction parameter
  # 
  #   eta <- exp(-delta * gamma)
  # 
  #   O <- diag(c(1, eta)) # Partial velocity refreshment: bit we keep
  #   B <- diag(c(1, 1)); B[2,1] <- -0.5 * delta / sig_sq # Kick
  #   A <- diag(c(1, 1)); A[1,2] <- delta # Drift
  # 
  #   BAB <- B %*% A %*% B
  #   
  #   return("slope" = BAB %*% O,
  #          "intercept_var" = BAB %*% diag(c(0, delta^2 * (1 - eta))) %*% BAB) # Partial velocity refreshment: variance of noise increment, after BAB steps
  # }
  
  # # Tracking the mean and variance
  # mu_t <- lapply(seq_along(mu_0), function(i) c(mu_0[i], 0))
  # Sigma_t <- lapply(seq_along(mu_0), function(i) diag(Sigma_0_diag[i], 1))
  # 
  # # Recursion
  # 
  # lapply(seq_along(mu_t), function(i) slope[[i]] %*% mu_t[[i]])
  # lapply(seq_along(Sigma_t), function(i) {slope[[i]] %*% Sigma_t[[i]] %*% t(slope[[i]]) + intercept_var[[i]]})
  # 
  # get_w2sq_coord <- function(i) abs(mu_t) mu_t, Sigma_t) abs(mu_t - mu_infty)
  
  # Stationary covariance matrix, i.e the biased one
  Sigma_infty_diag <- Sigma_target_diag / (1 - 0.25 * delta^2 / Sigma_target_diag) 
  
  d <- length(mu_0)
  iterations <- seq(0,iter,thin)
  
  w2sq_coordwise <-
    foreach(coord = 1:d, .combine = rbind) %do% {
      
      # Set up the recursion
      mu_diff    <- c(mu_0[coord] - mu_infty[coord], 0)
      Sigma_diff <- diag(c(Sigma_0_diag[coord] - Sigma_infty_diag[coord], 0))

      # Slope matrix
      B <- obab_slope_matrix(Sigma_target_diag[coord])
      Bthin <- expm::`%^%`(B, thin)

      # Recursion
      mu_diffs <- rep(NA, length(iterations))
      sigma_diffs <- rep(NA, length(iterations))

      for(it in iterations){
        # Update the mean and  covariance
        if(it > 0){
          mu_diff <- Bthin %*% mu_diff
          Sigma_diff <- Bthin %*% Sigma_diff %*% t(Bthin)
        }

        # Store the mean and standard deviation for the position
        mu_diffs[it/thin + 1] <- mu_diff[1]
        sigma_diffs[it/thin + 1] <- sqrt(Sigma_diff[1,1] + Sigma_infty_diag[coord]) - sqrt(Sigma_infty_diag[coord])
      }

      # Squared Wasserstein distance, univariate Gaussian case
      w2sqs <- mu_diffs^2 + sigma_diffs^2
      return(w2sqs)
    }

  w2sq <- data.frame("w2sq" = colSums(w2sq_coordwise), "iter" =  iterations)
  return(w2sq)
}



# # Tracking the mean and variance
# mu_t <- lapply(seq_along(mu_0), function(i) c(mu_0[i], 0))
# Sigma_t <- lapply(seq_along(mu_0), function(i) diag(Sigma_0_diag[i], 1))
# 
# # Recursion
# 
# lapply(seq_along(mu_t), function(i) slope[[i]] %*% mu_t[[i]])
# lapply(seq_along(Sigma_t), function(i) {slope[[i]] %*% Sigma_t[[i]] %*% t(slope[[i]]) + intercept_var[[i]]})
# 
