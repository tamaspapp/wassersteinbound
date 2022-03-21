
#####
# Empirical bounds
#####

#' Compute W2^2 between reference measure and list of empirical measures,
#' with jackknife variance estimates. (In parallel.)
#'
#' x = list of matrices, each matrix represents one empirical measure
#' 
#' x_reference = matrix representing "reference" empirical measure
#' 
#' R = number of samples in each empirical measure
#' 
#' ncores = number of cores to use
#'
#' @export 
EvaluateW2sqJackMCMC <- function (x, x_reference, R, ncores) {
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  w2sq_out <- foreach::foreach(x_ = x, .packages = "wassersteinbound") %dopar% {
    wassersteinbound::Flapjack(wassersteinbound::EvaluateSquaredCost(x_, x_reference))
  }
  
  parallel::stopCluster(cl)
  
  # Convert output to:
  #
  # (a) A vector of W2^2 costs;
  # (b) A matrix of jackknife W2^2 costs. Each column is associated to one MCMC iteration.
  
  w2sq <- rep(0, length(x))
  jack_w2sq <- matrix(nrow = R, ncol = length(x))
  
  for (i in 1:length(w2sq)) {
    w2sq[i] <- w2sq_out[[i]]$transp_cost
    jack_w2sq[, i] <- w2sq_out[[i]]$jack_data
  }
  
  return(list("w2sq" = w2sq, "jack_w2sq" = jack_w2sq))
}

#' Compute W2^2 between reference measure and list of empirical measures.
#' (In parallel.)
#'
#' x = list of matrices, each matrix represents one empirical measure
#' 
#' x_reference = matrix representing "reference" empirical measure
#' 
#' R = number of samples in each empirical measure
#' 
#' ncores = number of cores to use
#'
#' @export 
EvaluateW2sqMCMC <- function (x, x_reference, R, ncores) {
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  w2sq <- foreach::foreach(x_ = x, .packages = "wassersteinbound", .combine = "c") %dopar% {
    wassersteinbound::SolveAssignmentNetworkflow(wassersteinbound::EvaluateSquaredCost(x_, x_reference))
  }
  
  parallel::stopCluster(cl)
  
  return(list("w2sq" = w2sq))
}

#####
# Coupling bound computation
#####

#' Assemble W2 coupling bound.
#' 
#' R = number of chains
#' 
#' tau = vector of meeting times (measured at X-chain)
#' 
#' L = time lag
#' 
#' bound_comp_list = list of R vectors, each vector has entries 
#' ||X_{t+L} - Y_t||^2
#'
#' thin = thinning factor for the bound
#'
#' @export 
AssembleW2CouplingBound <- function(bound_comp_list,
                                    tau,
                                    L, 
                                    R,
                                    thin) {
  tau_max <- max(tau)      # Always L + 1 or more
  tau_y_max <- tau_max - L
  
  num <- floor((tau_y_max - 1) / thin) + 1
  
  # First, compute sqrt(E[||X_{t+L} - Y_t||^2]) for each t.
  
  w2_bound_components <- rep(0, num) # Must also include t = 0
  
  for (t in seq.int(0, tau_y_max - 1, thin)) {
    rs_notcoupled <- which(tau > (t + L)) # "Which Y-chains have not coupled by their time t?"
    
    # Sum the ||X_{t+L} - Y_t||^2, only for the chains that haven't yet coupled
    for (r in rs_notcoupled) {
      w2_bound_components[t/thin + 1] <- w2_bound_components[t/thin + 1] + bound_comp_list[[r]][t/thin + 1] 
    }
  }
  
  # Average, then square-root, to get on the right scale
  w2_bound_components <- sqrt(w2_bound_components / R)
  
  # Accumulate these components in jumps of L, with thinning
  w2_bound <- rep(NA, num)
  for (t in seq.int(0, tau_y_max - 1, thin)) {
    k <- pmax(0, floor((tau_y_max - 1 - t) / L)) # Number of additional bound components
    w2_bound[t/thin + 1] <- sum(w2_bound_components[t/thin + 1 + L/thin * (k > 0) * 0:k])
  }
  
  return(w2_bound)
}

#' Assemble TVD coupling bound from meeting times "tau" at X-chain
#' 
#' tau = vector of meeting times
#' 
#' L = time lag
#' 
#' R = number of replicates
#' 
#' thin = thinning factor for the bound
#'
#' @export
AssembleTVDCouplingBound <- function(tau, L, R, thin) {
  tau_y_max <- max(tau) - L
  
  num <- floor((tau_y_max - 1) / thin) + 1
  
  bound_components <- bound <- rep(0, num)
  
  # Compute bound components at each iteration t: mean(Indicator(Y_t != X_{t+L}))
  for (t in seq.int(0, tau_y_max - 1, thin)) {
    bound_components[t/thin + 1] <- sum(tau > (t + L)) / R
  }
  
  # Accumulate bound components in jumps of L, with thinning
  for (t in seq.int(0, tau_y_max - 1, thin)) {
    k <- pmax(0, floor((tau_y_max - 1 - t) / L)) # Number of additional bound components
    
    bound[t / thin + 1] <- sum(bound_components[t / thin + 1 + L/thin * (k > 0) * 0:k])
  }
  
  return(bound)
}

#####
# Target parameters
#####

#' Generate target variance matrix for periodic boundary AR(1) experiment.
#' 
#' d = dimension
#' 
#' alpha = correlation in AR(1) structure
#' 
#' sigma = standard deviation of noise terms
#'
#' @export 
GenerateTargetCovarianceMatrix_ar1 <- function(d, alpha, sigma) {
  Sigma <- matrix(NA, ncol = d, nrow = d)
  
  if(d == 1) {
    Sigma[1, 1] <- sigma^2 / (1 - alpha^2)
  } else {
    
    const <- sigma^2 / (1 - alpha^d) / (1 - alpha^2)
    
    for(i in 1:d){
      for(j in 1:d){
        k <- abs(i - j)
        Sigma[i, j] <- Sigma[j, i] <- alpha^k + alpha^(d - k)
      }
    }
    Sigma <- const * Sigma 
  }
  
  return(Sigma)
}

#' Generate target variance matrix (Sigma) and its Cholesky factor (upper 
#' triangular; U) for the MALA vs ULA dimensional scaling experiment
#'
#' Matrices are sparse, stored in Matrix::dgCMatrix format.
#'
#' @export 
GenerateSigmaAndU_scaling <- function(d) {
  if (d == 1) {
    Sigma_inv <- as(matrix(1), "dgCMatrix")
    U <- as(matrix(1), "dgCMatrix")
  } else {
    # Sparse matrix: inverse target covariance
    Sigma_inv <- Matrix::Matrix(0, nrow = d, ncol = d, sparse = TRUE) 
    # Sparse matrix: upper triangular part of Cholesky factorization of Sigma_inv
    U <- Matrix::Matrix(0, nrow = d, ncol = d, sparse = TRUE)
    
    diag(Sigma_inv) <- 5/3
    Sigma_inv[1, 1] <- Sigma_inv[d, d] <- 4/3
    Sigma_inv[abs(row(Sigma_inv) - col(Sigma_inv)) == 1] <- -2/3
    
    diag(U) <- 2/sqrt(3)
    U[col(U) - row(U) == 1] <- -1/sqrt(3)
    U[d, d] <- 1
  }
  
  return(list("Sigma_inv" = Sigma_inv, "U" = U))
}

#####
# Miscellaneous
#####

#' Density for mixture of Gaussians
#'
#' x = points to be evaluated at
#'
#' ps = mixture weights, automatically normalized to sum to 1
#'
#' mus = means of mixture components
#' 
#' sigmas = standard deviations of mixture components
#'
#' @export
GaussianMixtureDensity <- function(x, ps, mus, sigmas)
{
  ps <- ps / sum(ps)
  
  func <- function(x, ps, mus, sigmas)
  {
    return(sum(ps * dnorm(x, mean = mus, sd = sigmas)))
  }
  
  func_vec <- Vectorize(function(x){ func(x, ps, mus, sigmas) }, "x")
  
  return(func_vec(x))
}


