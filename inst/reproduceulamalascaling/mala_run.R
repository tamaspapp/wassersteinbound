library(wassersteinbound)
library(doParallel)
library(Matrix)

# Parameters
source("params.R")

H <- 1 / D^(1/6)

disp <- sqrt(3)

start_debias <- 250L
iter         <- 500L
L            <- 2000L
iter_final   <- L
thin         <- 1L

thin_cpl <- 1L

end_debias   <- iter
debias_iters <- seq(floor(start_debias/ thin) + 1, floor(end_debias/ thin) + 1)

mala <- vector("list", length(D))

for(k in 1:length(D)) {
  
  d <- D[k]
  h <- H[k]
  
  mats <- GenerateSigmaAndU_scaling(d)
  
  Sigma_inv <- as(mats$Sigma_inv, "dgCMatrix")
  U <- as(mats$U, "dgCMatrix")
  
  
  # Do MCMC in parallel
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  in_time <- Sys.time()
  out <- foreach(i = 1:R, .packages = "wassersteinbound") %dopar% {
    SetSeed_cpp(seed, i)
    mcmc <- SimulateReflMaxMALA_scaling(Sigma_inv, U, disp, h, L, iter, iter_final, thin)
    
    return(mcmc)
  }
  
  out_time <- Sys.time()
  print(out_time - in_time)
  
  parallel::stopCluster(cl)
  
  #####
  # Format data for convenience
  
  tau <- rep(NA, R)

  for (i in 1:R) {
    tau[i] <- out[[i]]$tau
  }

  x <- vector(mode = "list", length = floor(iter / thin) + 1)


  a <- matrix(NA, nrow = d, ncol = R)

  for (i in 1:length(x)) {
    for (j in 1:R) {
      a[, j] <- out[[j]]$x[[i]]
    }

    x[[i]] <- a
  }

  x_reference <- matrix(NA, nrow = d, ncol = R)

  for (j in 1:R) {
    x_reference[, j] <- out[[j]]$x_reference
  }

  w2_bound_components <- vector(mode = "list", length = R)

  for (i in 1:R) {
    w2_bound_components[[i]] <- out[[i]]$w2_bound_parts
  }
  
  rm(out, a)

  # Get coupling bounds
  tvd_coupling_bound <- AssembleTVDCouplingBound(tau, L, R, thin_cpl)
  w2_coupling_bound <- AssembleW2CouplingBound(
    w2_bound_components,
    tau, L, R, thin_cpl
  )

  # Evaluate empirical W2^2, with jackknife leave-one-out estimates

  in_time <- Sys.time()
  w2sq_out <- EvaluateW2sqMCMC(x, x_reference, R, ncores)
  out_time <- Sys.time()
  print(out_time - in_time)

  w2sq <- w2sq_out$w2sq

  #####
  # Debias to obtain empirical W2^2 bounds, and compute confidence intervals

  source("get_bounds.R")

  #####
  # Save output

  mala[[k]] <- list(
    "w2sq" = w2sq,
    "w2sq_ub" = w2sq_ub,
    "w2sq_lb" = w2sq_lb,
    "w2_lb" = w2_lb,
    "tau" = tau,
    "w2_bound_components" = w2_bound_components,
    "w2_coupling_bound" = w2_coupling_bound,
    "tvd_coupling_bound" = tvd_coupling_bound
  )
  
  rm(tau, w2_bound_components, x, x_reference)
}

for (i in 1:9) {
  mala[[i]]$w2_bound_components <- NULL
}

save(mala, file = "mala.Rdata")

