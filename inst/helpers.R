library(doParallel)
library(scales)
library(dplyr)

rbind_within_list <- function(x, ...) Map(function(...) rbind(...), x, ...)
# rbind_within_list <- function(x, ...) Map(function(...) do.call("rbind", list(...)), x, ...) # f(A, B) = list(rbind(A[[1]], B[[1]]), rbind(A[[2]], B[[2]]), ...)

concatenate_within_list <- function(x, ...) Map(function(...) c(...), x, ...) # f(A, B) = list(c(A[[1]], B[[1]]), c(A[[2]], B[[2]]), ...)

get_singlechain_output <- function(mcmc_algorithm, target_parameters, mcmc_parameters, x0s, iter, thin, ncores, seed) {
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)

  R <- length(x0s)

  output <- foreach(r = 1:R, x0 = x0s,
                    .combine = rbind_within_list, .multicombine = T,
                    .packages = c("wassersteinbound", "Matrix")) %dopar% {
    SetSeed_pcg32(seed, r)
    return(asplit(mcmc_algorithm(target_parameters, mcmc_parameters, x0, iter, thin)$xs, 1))
  }
  parallel::stopCluster(cl)
  return(output)
}

get_empirical_w2sq <- function(singlechain_out, x_ref, ncores) {
  fix_zeros <- function(x) ifelse(x > 0, x, 0) # Deal with negative values due to floating point error
  
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  w2sq_empirical_estimates <-
    foreach(x_ = singlechain_out, .combine = concatenate_within_list, .multicombine = T, .packages = "wassersteinbound") %dopar% {
      
      out <- w2sq_empirical(x_ref, x_)
      
      return(list("w2sq" = fix_zeros(out$w2sq), 
                  "potentials_xref" = list(out$potentials_x),
                  "potentials_xt" = list(out$potentials_y),
                  "consistent_stderr" = sqrt(var(out$potentials_x + out$potentials_y)/ nrow(x_))))
    }
  parallel::stopCluster(cl)

  return(w2sq_empirical_estimates)
}

get_empirical_bounds <- function(w2sq_empirical_estimates, iters, debias, conf_level = 0.95){
  signed_square <- function(x){ sign(x) * x^2 }
  
  # Z-values for Gaussian confidence intervals
  z_up <- qnorm(1 - (1-conf_level)/2)
  z_lo <- qnorm((1-conf_level)/2)
  
  # Debias
  w2sq_est <- w2sq_empirical_estimates$w2sq
  debias_range <- (iters >= debias[1]) & (iters <= debias[2])
  w2sq_debias <- mean(w2sq_est[debias_range])
  
  w2_est <- sqrt(w2sq_est)
  w2_debias <- mean(w2_est[debias_range])
  
  mean_reduction <- function(x) Reduce("+", x) / length(x)
  stderr <- function(x) sqrt(var(x) / length(x))
  
  # Calculate consistent variance estimator for W2sq
  get_stderrs_w2sq <- function(empirical, debias_range){
    
    n <- length(empirical$potentials_xt[[1]]) # Sample size
    
    # Uncorrected plug-in W2sq estimator
    potentials_main <- Map("+", empirical$potentials_xt, empirical$potentials_xref)
  
    # Debiasing term for plug-in W2sq estimator
    potentials_xt_debias <- mean_reduction(empirical$potentials_xt[debias_range])
    potentials_xref_debias <- mean_reduction(empirical$potentials_xref[debias_range])
    potentials_debias <- potentials_xt_debias + potentials_xref_debias
    
    # Subtract potentials corresponding to debiasing term. 
    potentials_aggregated <- Map("-", potentials_main, list(potentials_debias)) # Putting the second expression into a list is essential!
    
    return(sapply(potentials_aggregated, stderr))
  }
  
  # Calculate consistent (heuristic) variance estimator for W2
  get_stderrs_w2 <- function(empirical, debias_range){
    
    n <- length(empirical$potentials_xt[[1]]) # Sample size
    
    # Uncorrected plug-in W2 estimator
    potentials_main <- Map("+", empirical$potentials_xt, empirical$potentials_xref)
    pot_div_main <- Map("/", potentials_main, as.list(2*sqrt(empirical$w2sq)))
    
    # Debiasing term for plug-in W2 estimator
    potentials_debias <- Map("+", empirical$potentials_xt[debias_range], empirical$potentials_xref[debias_range])
    pot_div_debias <- mean_reduction(Map("/", potentials_debias, as.list(2*sqrt(empirical$w2sq[debias_range]))))
    
    # Subtract
    pot_div_aggregated <- Map("-", pot_div_main, list(pot_div_debias)) # Putting the second expression into a list is essential!
    
    return(sapply(pot_div_aggregated, stderr))
  }
  
  w2sq_stderr <- get_stderrs_w2sq(w2sq_empirical_estimates, debias_range)
  w2_stderr <- get_stderrs_w2(w2sq_empirical_estimates, debias_range)

  # U
  u_df <- data.frame("iter"  = iters, 
                     "w2sq"  = w2sq_est - w2sq_debias,
                     "ci_up" = w2sq_est - w2sq_debias + z_up * w2sq_stderr,
                     "ci_lo" = w2sq_est - w2sq_debias + z_lo * w2sq_stderr,
                     "estimator" = "U")
  # L
  l_df <- data.frame("iter"  = iters, 
                     "w2sq"  = signed_square(w2_est - w2_debias),
                     "ci_up" = signed_square(w2_est - w2_debias + z_up * w2_stderr),
                     "ci_lo" = signed_square(w2_est - w2_debias + z_lo * w2_stderr),
                     "estimator" = "L")
  
  empirical_df <- rbind(u_df, l_df)
  return(empirical_df)
}

get_coupling_bound <- function(coupling_out, iters, name = "Coupling", conf_level = 0.95, boot_reps = 1e3){
  # N out of N bootstrap
  boot_colMeans <- function(df_, reps){
    n <- nrow(df_)
    boot <- lapply(1:reps, function(x) colMeans(df_[sample.int(n,replace = T),]))
    boot <- do.call(rbind, boot)
    return(boot)
  }
  
  df <- do.call(rbind, coupling_out)
  w2sq_est <- colMeans(df)
  boot_out <- boot_colMeans(df, boot_reps)
  
  u_up <- (1 + conf_level)/2
  u_lo <- (1 - conf_level)/2
  
  out_df <- data.frame("iter"  = iters, 
                       "w2sq"  = w2sq_est,
                       "ci_up" = apply(boot_out, 2, function(x)quantile(x, u_up)),
                       "ci_lo" = apply(boot_out, 2, function(x)quantile(x ,u_lo)),
                       "estimator" = name)
  return(out_df)
}

# Custom axis transformation for gggplot2: log(const + x)
log.const.p <- function(const = 0.1, name = "log.const.p") {
  scales::trans_new(name, 
                    function(x) {log10(x + const)},
                    function(y) {10^(y) - const},
                    domain = c(-const,Inf))
}

