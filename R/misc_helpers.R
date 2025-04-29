
#' "n-out-of-n" boostrap variance estimate from R repeats
#' 
#' x = vector, data to be resampled
#' R = number of bootstrap replicates
#'
#' @export
BoostrapVarianceBasic <- function(x, R) {
  n <- as.integer(length(x))
  out <- rep(NA, R)
  
  for (i in 1:R) {
    indices <- sample.int(n, replace = TRUE)
    out[i] <- var(x[indices])
  }
  return(out)
}


#' AR(1) process target parameters.
#' 
#' Input:
#'  - d: dimension
#'  - rho: autocorrelation
#' 
#' Output:
#'  - Sigma: covariance matrix.
#'  - Omega: precision matrix, in sparse format.
#'  - Omega_chol_u: upper Choelsky factor of Omega.
#'  
#' @export
AR1_target_params <- function(d, rho = 0.5) {
  Sigma <- rho^outer(1:d, 1:d, FUN = function(i,j)abs(i-j))
  
  if (d == 1) {
    Omega <- as(matrix(1), "dgCMatrix")
    Omega_chol_u <- as(matrix(1), "dgCMatrix")
  } else {
    # Precision matrix, see: https://nhigham.com/2021/07/06/what-is-the-kac-murdock-szego-matrix/
    most_columns <- c(1 + rho^2, -rho, rep(0, d - 2))
    
    Omega <- as(toeplitz(most_columns), "dgCMatrix") # Have to change two entries and scale the entire thing
    Omega[1, 1] <- Omega[d, d] <- 1 # Fix the start and end
    Omega <- Omega / (1 - rho^2)
    
    # Its upper Cholesky factor
    Omega_chol_u <- as(chol(Omega), "dgCMatrix")
    Omega_chol_u <- Omega_chol_u
  }
  
  return(list("Sigma"        = Sigma,
              "Omega"        = Omega,
              "Omega_chol_u" = Omega_chol_u))
}



#' Format output for single-chain MCMC from list of R matrices of samples, where
#' each matrix is ("iter" x "d"), to a list of "iter" empirical measures, each 
#' stored as a matrix ("R" x "d").
#' 
#' @export
format_singlechain_out <- function(xs_list) {
  R <- length(xs_list)
  d <- ncol(xs_list[[1]])
  iter <- nrow(xs_list[[1]])
  
  out <- vector("list", iter)
  
  temp <- matrix(NA, nrow = R, ncol = d)
  for (it in 1:iter) {
    for(r in 1:R) {
      temp[r, ] <- xs_list[[r]][it, ]
    }
    out[[it]] <- temp
  }
  return(out)
}


#' Calculate empirical squared Wasserstein distance in the MCMC setting
#'
#' Calculate the squared empirical Wasserstein distance between hatpi_t and
#' hatpi_T for all t in 0,1,...T, from a list of empirical measures
#' list(hatpi_0, ..., hatpi_T).
#'
#' @export
get_empirical_w2sq_mcmc <- function(mcmc_out, ncores, ref_from_mcmc = T, x_ref = NA) {

  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  if(ref_from_mcmc) {
    x_ref <- mcmc_out[[length(mcmc_out)]]
  }
  
  w2sq_empirical_estimates <-
    foreach(x_ = mcmc_out, .combine = rbind, .packages = "wassersteinbound") %dopar% {
  
      out <- w2sq_empirical(x_, x_ref)
  
      n <- nrow(x_)
      data.frame("w2sq" = out$w2sq,
                 "naive_stderr" = sqrt(var(out$assignment_cost_fractions) / n),
                 "consistent_stderr" = sqrt((var(out$potentials_x) + var(out$potentials_y))/ n))
    }
  names(w2sq_empirical_estimates) <- c("w2sq", "naive_stderr", "consistent_stderr")
  
  parallel::stopCluster(cl)
  
  return(w2sq_empirical_estimates)
}
