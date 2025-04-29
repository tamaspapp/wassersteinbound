################ Bayesian High dimensional regression functions ################
# Code modified from https://github.com/niloyb/BoundWasserstein

################ Single chain MCMC functions ################
# Functions for a blocked Gibbs sampler with half-t priors

## Matrix calculation helper functions ##
X_eta_tX <- function(eta, X, X_transpose){
  return(cpp_crossprod(X_transpose*c(1/eta)^0.5))
}
M_matrix <- function(xi, eta, X_eta_tX_matrix, n){
  if(length(eta)==0) return(diag(n))
  return(diag(n) + (xi^-1)*X_eta_tX_matrix)
}

## xi update given eta ##
# Unnormalized posterior pdf of log(xi)
log_ratio <- function(xi, eta, X_eta_tX_matrix, y, a0, b0)
{
  n <- length(y)
  M <- M_matrix(xi,eta,X_eta_tX_matrix,n)
  chol_M <- chol(M)
  log_det_M <- 2*sum(log(diag(chol_M)))
  M_inverse <- chol2inv(chol_M)
  ssr <- b0 + t(y)%*%((M_inverse)%*%y)
  log_likelihood <- -0.5*log_det_M -0.5*(n+a0)*log(ssr)
  log_prob <- -log(sqrt(xi)*(1+xi))
  return(list('log_likelihood'=log_likelihood+log_prob,
              'ssr' = ssr, 'M_matrix_inverse' = M_inverse))
}
# Unnormalized posterior pdf of log(xi) for approximate MCMC
log_ratio_approx <- function(xi, eta, X, X_transpose, y, a0, b0, active_set)
{
  n <- length(y)
  log_prob <- -log(sqrt(xi)*(1+xi))
  if (sum(active_set)==0)
  {
    M_inverse <- diag(n)
    ssr <- b0 + sum(y^2)
    log_likelihood <- -0.5*(n+a0)*log(ssr)
  } else{
    eta <- eta[active_set]
    X <- X[,active_set,drop=F]
    X_transpose <- X_transpose[active_set,,drop=F]
    if(sum(active_set)==1){
      woodbury_matrix_part <- xi*eta + cpp_crossprod(X) #cpp_prod(X_transpose, X)
    } else{
      woodbury_matrix_part <- xi*diag(eta) + cpp_crossprod(X)# cpp_prod(X_transpose, X)
    }
    woodbury_matrix_part_inverse <- chol2inv(chol(woodbury_matrix_part))
    M_inverse <- diag(n) -
      cpp_prod(cpp_prod(X,woodbury_matrix_part_inverse),X_transpose)
    log_det_M <- sum( log( xi^(-1)*(svd((eta^(-0.5))*X_transpose)$d)^2 + 1 ) )
    ssr <- b0 + t(y)%*%((M_inverse)%*%y)
    log_likelihood <- -0.5*log_det_M -0.5*(n+a0)*log(ssr)
  }
  return(list('log_likelihood'=log_likelihood+log_prob, 'ssr' = ssr,
              'M_matrix_inverse' = M_inverse))
}
# Metropolis-Hastings update of xi given eta
xi_update <- function(current_xi, eta, X, X_transpose, y, a0, b0, std_MH,
                      approximate_algo_delta=0, fixed=FALSE)
{
  n <- length(y)
  if (fixed==TRUE){
    min_xi <- current_xi
    active_set <- ((min_xi*eta)^(-1) > approximate_algo_delta)
    
    # Matrix calculations
    if (sum(active_set)>n) # Woodbury inversion when sum(active_set)>n
    {
      X_eta_tX_matrix <- X_eta_tX(eta[active_set], X[ ,active_set,drop=F],
                                  X_transpose[active_set, ,drop=F])
      log_ratio_current_and_ssr <- log_ratio(current_xi, eta[active_set],
                                             X_eta_tX_matrix, y, a0, b0)
    } else { # Woodbury inversion when sum(active_set)<n
      log_ratio_current_and_ssr <-
        log_ratio_approx(current_xi, eta, X, X_transpose, y, a0, b0, active_set)
    }
  } else {
    proposed_xi <- exp(rnorm(1, log(current_xi), std_MH))
    min_xi <- min(current_xi, proposed_xi)
    active_set <- ((min_xi*eta)^(-1) > approximate_algo_delta)
    
    # Matrix calculations
    if (sum(active_set)>n) # Woodbury inversion when sum(active_set)>n
    {
      X_eta_tX_matrix <- X_eta_tX(eta[active_set],X[ ,active_set,drop=F],
                                  X_transpose[active_set, ,drop=F])
      log_ratio_current_and_ssr <- log_ratio(current_xi, eta[active_set],
                                             X_eta_tX_matrix, y, a0, b0)
      log_ratio_proposed_and_ssr <- log_ratio(proposed_xi, eta[active_set],
                                              X_eta_tX_matrix, y, a0, b0)
    } else { # Woodbury inversion when sum(active_set)<n
      log_ratio_current_and_ssr <-
        log_ratio_approx(current_xi, eta, X, X_transpose, y, a0, b0, active_set)
      log_ratio_proposed_and_ssr <-
        log_ratio_approx(proposed_xi, eta, X, X_transpose, y, a0, b0, active_set)
    }
    
    # MH accept- reject step
    log_accept_prob <-
      (log_ratio_proposed_and_ssr$log_likelihood -
         log_ratio_current_and_ssr$log_likelihood) +
      (log(proposed_xi)-log(current_xi))
    
    if (log(runif(1))<log_accept_prob)
    {
      current_xi <- proposed_xi
      log_ratio_current_and_ssr <- log_ratio_proposed_and_ssr
    }
  }
  return(list('xi'=current_xi, 'ssr' = log_ratio_current_and_ssr$ssr,
              'M_matrix' = log_ratio_current_and_ssr$M_matrix_inverse,
              'active_set' = active_set))
}

## sigma^2 update step given eta and xi
sigma2_update <- function(xi, eta, n, ssr, a0, b0, sigma2_fixed_value=NULL)
{
  if(!is.null(sigma2_fixed_value)) return(sigma2_fixed_value)
  return(1/(rgamma(1, shape = (n+a0)/2, rate = (ssr)/2)))
}

## beta update given xi, eta, sigma2
beta_update <- function(xi, sigma2, eta, X, X_transpose, y,
                        M_matrix_inverse, active_set)
{
  p <- length(eta)
  n <- length(y)
  u = rnorm(p, 0, 1)
  u = (sqrt(xi*eta)^-1)*u
  v = X%*%u + c(rnorm(n,0,1))
  v_star <- M_matrix_inverse%*%(y/sqrt(sigma2) - v)
  U <- (xi^-1)* ( ((eta[active_set])^(-1))*(X_transpose[active_set, ,drop=F]) )
  u[active_set] <- u[active_set] + U%*%v_star
  beta <- sqrt(sigma2)*(u)
  return(beta)
}


## Full blocked Gibbs samplers ##
half_t_kernel <-
  function(X, X_transpose, y, a0=1, b0=1, std_MH=0.8,
           xi_current, sigma2_current,
           beta_current, eta_current, approximate_algo_delta,
           nrepeats_eta = 1, verbose = FALSE,
           xi_fixed=FALSE, sigma2_fixed=FALSE, t_dist_df)
  {
    if (verbose) ptm <- proc.time()
    n <- dim(X)[1]
    p <- dim(X)[2]
    # xi update
    xi_new <- xi_update(xi_current, eta_current, X, X_transpose, y, a0, b0, std_MH,
                        approximate_algo_delta, fixed=xi_fixed)
    # sigma2 update
    sigma2_fixed_value <- NULL
    if(sigma2_fixed==TRUE) sigma2_fixed_value <- sigma2_current
    sigma2_new <- sigma2_update(xi_new$xi, eta_current, n, xi_new$ssr, a0, b0,
                                sigma2_fixed_value)
    # beta update
    beta_new <- beta_update(xi_new$xi, sigma2_new, eta_current, X, X_transpose, y,
                            xi_new$M_matrix, xi_new$active_set)
    # eta update
    eta_new <- eta_update_half_t(xi_new$xi, sigma2_new, beta_new, eta_current,
                                 t_dist_df, nrepeats_eta)
    
    if (verbose) print(proc.time()[3]-ptm[3])
    output <- list('beta_samples'=beta_new, 'eta_samples'=eta_new,
                  'sigma2_samples'=sigma2_new, 'xi_samples'=xi_new$xi,
                  "active_set"=xi_new$active_set)
    return(output)
  }


half_t_mcmc <-
  function(chain_length, burnin, X, X_transpose, y, a0=1, b0=1, std_MH=0.8,
           rinit=NULL, approximate_algo_delta=0, nrepeats_eta = 1,
           verbose = FALSE, xi_fixed=FALSE, sigma2_fixed=FALSE, t_dist_df,
           thin=1,store_all=T)
  {
    n <- dim(X)[1]
    p <- dim(X)[2]
    if(is.null(rinit)){
      # Initializing from the prior
      rinit <- function(){
        xi <- (1/rt(1, df=1))^2
        sigma2 <- 1/rgamma(1, shape = a0/2, rate = b0/2)
        eta <- (1/rt(p, df=t_dist_df))^2
        beta <- rnorm(p)*sqrt(sigma2/(xi*eta))
        return(list(xi = xi, sigma2 = sigma2, beta = beta, eta = eta))
      }
    }
    
    # Initializing
    chain <- rinit()
    
    xi_current <- chain$xi
    beta_current <- chain$beta
    sigma2_current <- chain$sigma2
    eta_current <- chain$eta
    
    rm(chain)
    
    # Setting up storage
    if(store_all) {
      xi_samples <- rep(NA, (chain_length-1)%/%thin+1)
      sigma2_samples <- rep(NA, (chain_length-1)%/%thin+1)
      beta_samples <- matrix(NA, nrow=(chain_length-1)%/%thin+1, ncol = p)
      eta_samples <- matrix(NA, nrow=(chain_length-1)%/%thin+1, ncol = p)
      active_sets <- matrix(F, nrow=(chain_length-1)%/%thin+1, ncol = p)
      
      if(burnin == 0) { # Store the initialization
        xi_samples[1] <- xi_current
        sigma2_samples[1] <- sigma2_current
        beta_samples[1,] <- beta_current
        eta_samples[1,] <- eta_current
      }
    } else {
      beta_samples <- matrix(NA, nrow=(chain_length-1)%/%thin+1, ncol = p)
      if(burnin == 0) beta_samples[1,] <- beta_current
    }
    
    # Doing MCMC
    if(chain_length + burnin > 1){ # Do at least one iteration
      for(iter in 1:(chain_length + burnin - 1))
      {
        output <-
          half_t_kernel(X, X_transpose, y, a0=a0, b0=b0, std_MH=std_MH,
                        xi_current, sigma2_current, beta_current, eta_current,
                        approximate_algo_delta, nrepeats_eta = nrepeats_eta,
                        verbose = verbose, xi_fixed=xi_fixed,
                        sigma2_fixed=sigma2_fixed, t_dist_df)
        xi_current <- output$xi_samples
        sigma2_current <- output$sigma2_samples
        beta_current <- output$beta_samples
        eta_current <- output$eta_samples
        active_set <- output$active_set
        
        # Start storage
        if((iter-burnin) %% thin == 0)
        {
          idx <- (iter-burnin)/thin + 1
          if(store_all) {
            xi_samples[idx] <- xi_current
            sigma2_samples[idx] <- sigma2_current
            beta_samples[idx,] <- beta_current
            eta_samples[idx,] <- eta_current
          } else {
            beta_samples[idx,] <- beta_current
          }
        } # End storage
      }
    }
    
    # Returning output
    if(store_all) return(list('beta_samples'=beta_samples, 'eta_samples'=eta_samples,
                              'sigma2_samples'=sigma2_samples, 'xi_samples'=xi_samples,
                              'active_sets'=active_sets))
    else return(list('beta_samples'=beta_samples))
  }

################ Coupled chain MCMC functions ################
# Functions for the coupled blocked Gibbs sampler with half-t priors

## Couplings for xi update given eta ##
# Coupled Metropolis-Hastings update of xi given eta
crn_xi_coupling <- 
  function(current_xi_1, eta_1, current_xi_2, eta_2,
           X, X_transpose, y, a0, b0, std_MH,
           approximate_algo_delta_1=0, approximate_algo_delta_2=0,
           fixed=FALSE)
{
  n <- length(y)
  
  if(fixed==TRUE){ # When xi is fixed in the Gibbs sampler
    min_xi_1 <- current_xi_1
    min_xi_2 <- current_xi_2
    active_set_1 <- ((min_xi_1*eta_1)^(-1) > approximate_algo_delta_1)
    active_set_2 <- ((min_xi_2*eta_2)^(-1) > approximate_algo_delta_2)
    
    if (sum(active_set_1)>n)
    {
      X_eta_tX_matrix_1 <-
        X_eta_tX(eta_1[active_set_1],X[, active_set_1, drop=F],
                 X_transpose[active_set_1, , drop=F])
      log_ratio_current_ssr_matrixinv_1 <-
        log_ratio(current_xi_1, eta_1[active_set_1], X_eta_tX_matrix_1, y, a0, b0)
    } else {
      log_ratio_current_ssr_matrixinv_1 <-
        log_ratio_approx(current_xi_1, eta_1, X, X_transpose, y, a0, b0, active_set_1)
    }
    
    if (sum(active_set_2)>n)
    {
      X_eta_tX_matrix_2 <-
        X_eta_tX(eta_2[active_set_2],X[, active_set_2, , drop=F], X_transpose[active_set_2, , drop=F])
      log_ratio_current_ssr_matrixinv_2 <-
        log_ratio(current_xi_2, eta_2[active_set_2], X_eta_tX_matrix_2, y, a0, b0)
    } else {
      log_ratio_current_ssr_matrixinv_2 <-
        log_ratio_approx(current_xi_2, eta_2, X, X_transpose, y, a0, b0, active_set_2)
    }
  } else { # When xi is varying in the Gibbs sampler
    standard_normal <- rnorm(1, mean = 0, sd = 1)
    # using common random numbers to get the proposal in the MH-algo
    log_proposed_xi_1 <- standard_normal*sqrt(std_MH) + log(current_xi_1)
    log_proposed_xi_2 <- standard_normal*sqrt(std_MH) + log(current_xi_2)
    
    proposed_xi_1 <- exp(log_proposed_xi_1)
    proposed_xi_2 <- exp(log_proposed_xi_2)
    
    min_xi_1 <- min(current_xi_1, proposed_xi_1)
    min_xi_2 <- min(current_xi_2, proposed_xi_2)
    active_set_1 <- ((min_xi_1*eta_1)^(-1) > approximate_algo_delta_1)
    active_set_2 <- ((min_xi_2*eta_2)^(-1) > approximate_algo_delta_2)
    
    if (sum(active_set_1)>n)
    {
      X_eta_tX_matrix_1 <- X_eta_tX(eta_1[active_set_1],X[, active_set_1, drop=F], X_transpose[active_set_1, , drop=F])
      log_ratio_current_ssr_matrixinv_1 <- log_ratio(current_xi_1, eta_1[active_set_1], X_eta_tX_matrix_1, y, a0, b0)
      log_ratio_proposed_ssr_matrixinv_1 <- log_ratio(proposed_xi_1, eta_1[active_set_1], X_eta_tX_matrix_1, y, a0, b0)
    } else {
      log_ratio_current_ssr_matrixinv_1 <- log_ratio_approx(current_xi_1, eta_1, X, X_transpose, y, a0, b0, active_set_1)
      log_ratio_proposed_ssr_matrixinv_1 <- log_ratio_approx(proposed_xi_1, eta_1, X, X_transpose, y, a0, b0, active_set_1)
    }
    
    if (sum(active_set_2)>n)
    {
      X_eta_tX_matrix_2 <- X_eta_tX(eta_2[active_set_2],X[, active_set_2, , drop=F], X_transpose[active_set_2, , drop=F])
      log_ratio_current_ssr_matrixinv_2 <- log_ratio(current_xi_2, eta_2[active_set_2], X_eta_tX_matrix_2, y, a0, b0)
      log_ratio_proposed_ssr_matrixinv_2 <- log_ratio(proposed_xi_2, eta_2[active_set_2], X_eta_tX_matrix_2, y, a0, b0)
    } else {
      log_ratio_current_ssr_matrixinv_2 <- log_ratio_approx(current_xi_2, eta_2, X, X_transpose, y, a0, b0, active_set_2)
      log_ratio_proposed_ssr_matrixinv_2 <- log_ratio_approx(proposed_xi_2, eta_2, X, X_transpose, y, a0, b0, active_set_2)
    }
    
    log_u <- log(runif(1))
    
    log_accept_prob_1 <- (log_ratio_proposed_ssr_matrixinv_1$log_likelihood - log_ratio_current_ssr_matrixinv_1$log_likelihood) + (log(proposed_xi_1)-log(current_xi_1))
    if (log_u<log_accept_prob_1){
      current_xi_1 <- proposed_xi_1
      log_ratio_current_ssr_matrixinv_1 <- log_ratio_proposed_ssr_matrixinv_1
    }
    
    log_accept_prob_2 <- (log_ratio_proposed_ssr_matrixinv_2$log_likelihood - log_ratio_current_ssr_matrixinv_2$log_likelihood) + (log(proposed_xi_2)-log(current_xi_2))
    if (log_u<log_accept_prob_2){
      current_xi_2 <- proposed_xi_2
      log_ratio_current_ssr_matrixinv_2 <- log_ratio_proposed_ssr_matrixinv_2
    }
  }
  return(list('xi_values'=c(current_xi_1, current_xi_2), 
              'log_ratio_ssr_matrix_inv_1'=log_ratio_current_ssr_matrixinv_1,  
              'log_ratio_ssr_matrix_inv_2'=log_ratio_current_ssr_matrixinv_2, 
              'active_set_1'= active_set_1, 
              'active_set_2'= active_set_2))
}

## Couplings for sigma^2 update given eta ##
digamma <- function(x, alpha, beta){
  return(alpha * log(beta) - lgamma(alpha) - (alpha+1) * log(x) - beta / x)
}
rigamma <- function(n, alpha, beta){
  return(1/rgamma(n = n, shape = alpha, rate = beta))
}
rigamma_coupled <- function(alpha1, alpha2, beta1, beta2){
  x <- rigamma(1, alpha1, beta1)
  if (digamma(x, alpha1, beta1) + log(runif(1)) < digamma(x, alpha2, beta2)){
    return(c(x,x))
  } else {
    reject <- TRUE
    y <- NA
    while (reject){
      y <- rigamma(1, alpha2, beta2)
      reject <- (digamma(y, alpha2, beta2) + log(runif(1)) < digamma(y, alpha1, beta1))
    }
    return(c(x,y))
  }
}

### sigma^2 CRN coupling update step given eta and xi
sigma2_update_crn <- function(xi_1, eta_1, xi_2, eta_2, n, ssr_1, ssr_2, a0, b0,
                              sigma2_fixed_value=NULL)
{
  if(!is.null(sigma2_fixed_value)){
    sample <- c(1/sigma2_fixed_value, 1/sigma2_fixed_value)
  } else {
    # Common random number gamma draw
    crn_gamma <-  rgamma(1, shape = (n+a0)/2, rate = 1)
    sample <- crn_gamma / c((ssr_1/2), (ssr_2/2))
  }
  return(1/sample)
}
# Coupled update of sigma2 given eta
crn_sigma2_coupling <- function(xi_1, eta_1, xi_2, eta_2, n, ssr_1, ssr_2, a0, b0,
                                sigma2_fixed_value=NULL){
  output <- sigma2_update_crn(xi_1, eta_1, xi_2, eta_2, n, ssr_1, ssr_2, a0, b0, sigma2_fixed_value)
  return(output)
}

## Common random numbers coupling of beta given xi, eta, sigma2 ##
crn_joint_beta_update <-
  function(xi_1, sigma2_1, eta_1, xi_2, sigma2_2, eta_2,
           X, X_transpose, y, M_matrix_inverse_1, M_matrix_inverse_2,
           active_set_1, active_set_2)
  {
    n <- dim(X)[1]
    p <- dim(X)[2]
    # Using same common random numbers for draws on two chains
    random_u <- rnorm(p, 0, 1)
    random_delta <- c(rnorm(n,0,1))
    u_1 = (sqrt(xi_1*eta_1)^-1)*random_u
    v_1 = X%*%u_1 + random_delta
    v_star_1 <- M_matrix_inverse_1%*%(y/sqrt(sigma2_1) - v_1)
    if(sum(active_set_1)>0){
      U_1 = (xi_1^-1)*((eta_1[active_set_1]^(-1))*(X_transpose[active_set_1, ,drop=F]))
      u_1[active_set_1] <- u_1[active_set_1] + U_1%*%v_star_1
    }
    beta_parameter_1 <- sqrt(sigma2_1)*(u_1)
    u_2 = (sqrt(xi_2*eta_2)^-1)*random_u
    v_2 = X%*%u_2 + random_delta
    v_star_2 <- M_matrix_inverse_2%*%(y/sqrt(sigma2_2) - v_2)
    if(sum(active_set_2)>0){
      U_2 = (xi_2^-1)*((eta_2[active_set_2]^(-1))*(X_transpose[active_set_2, ,drop=F]))
      u_2[active_set_2] <- u_2[active_set_2] + U_2%*%v_star_2
    }
    beta_parameter_2 <- sqrt(sigma2_2)*(u_2)
    return(cbind(beta_parameter_1, beta_parameter_2))
  }

## Full coupled blocked Gibbs samplers ##
# CRN coupling
coupled_half_t_kernel <-
  function(X, X_transpose, y, a0=1, b0=1, std_MH=0.8,
           xi_1_current, xi_2_current, sigma2_1_current, sigma2_2_current,
           beta_1_current, beta_2_current, eta_1_current, eta_2_current,
           approximate_algo_delta_1=0, approximate_algo_delta_2=0,
           nrepeats_eta=1,
           verbose = FALSE, xi_fixed=FALSE, sigma2_fixed=FALSE, t_dist_df)
  {
    n <- dim(X)[1]
    p <- dim(X)[2]
    
    if (verbose) ptm <- proc.time()
    
    # 1. Slice sample eta | rest
    if (is.infinite(nrepeats_eta)){
      stop("Number of slice sampling must be finite")
    } else {
      for (i in 1:nrepeats_eta) {
        eta_sample <-
          eta_update_half_t_crn_couple(xi_1_current, beta_1_current, eta_1_current, sigma2_1_current,
                                       xi_2_current, beta_2_current, eta_2_current, sigma2_2_current,
                                       t_dist_df)
        eta_1_current <- eta_sample[,1]
        eta_2_current <- eta_sample[,2]
      }
    }
    eta_1_new <- eta_sample[,1]
    eta_2_new <- eta_sample[,2]
    
    if (verbose) print(proc.time()[3]-ptm[3])
    
    # 2. Sample xi | rest
    if (verbose) ptm <- proc.time()
    xi_sample <-
      crn_xi_coupling(xi_1_current, eta_1_new, xi_2_current, eta_2_new,
                      X, X_transpose, y, a0, b0, std_MH,
                      approximate_algo_delta_1, approximate_algo_delta_2,
                      fixed=xi_fixed)
    xi_1_new <- xi_sample$xi_values[1]
    xi_2_new <- xi_sample$xi_values[2]
    if (verbose) print(proc.time()[3]-ptm[3])
    
    # 2. Sample sigma | rest
    if (verbose) ptm <- proc.time()
    sigma2_fixed_value <- NULL
    if(sigma2_fixed==TRUE){sigma2_fixed_value <- sigma2_1_current}
    sigma2_sample <-
      crn_sigma2_coupling(xi_1_new,eta_1_new,xi_2_new,eta_2_new,n,
                              (xi_sample$log_ratio_ssr_matrix_inv_1)$ssr,
                              (xi_sample$log_ratio_ssr_matrix_inv_2)$ssr,a0,b0,
                              sigma2_fixed_value)
    sigma2_1_new <- sigma2_sample[1]
    sigma2_2_new <- sigma2_sample[2]
    if (verbose) print(proc.time()[3]-ptm[3])
    
    # 3. Sample beta | rest
    if (verbose) ptm <- proc.time()
    M_inverse_1 <- (xi_sample$log_ratio_ssr_matrix_inv_1)$M_matrix_inverse
    M_inverse_2 <- (xi_sample$log_ratio_ssr_matrix_inv_2)$M_matrix_inverse
    active_set_1 <- xi_sample$active_set_1
    active_set_2 <- xi_sample$active_set_2
    beta_samples <- crn_joint_beta_update(xi_1_new, sigma2_1_new, eta_1_new,
                                          xi_2_new, sigma2_2_new, eta_2_new,
                                          X, X_transpose, y, M_inverse_1, M_inverse_2,
                                          active_set_1, active_set_2)
    beta_1_new <- beta_samples[,1]
    beta_2_new <- beta_samples[,2]
    if (verbose) print(proc.time()[3]-ptm[3])
    
    #chain1 <- list('beta'=beta_1_new, 'eta'=eta_1_new, 'sigma2'=sigma2_1_new, 'xi'=xi_1_new)
    #chain2 <- list('beta'=beta_2_new, 'eta'=eta_2_new, 'sigma2'=sigma2_2_new, 'xi'=xi_2_new)
    
    output <- list('beta_1_samples'=beta_1_new, 'beta_2_samples'=beta_2_new,
                   'eta_1_samples'=eta_1_new, 'eta_2_samples'=eta_2_new,
                   'sigma2_1_samples'=sigma2_1_new, 'sigma2_2_samples'=sigma2_2_new,
                   'xi_1_samples'=xi_1_new, 'xi_2_samples'=xi_2_new)
    
    return(output)
  }


#'
#' coupled_half_t_mcmc
coupled_half_t_mcmc <-
  function(burnin, chain_length, thin, X, X_transpose, y, a0=1, b0=1, std_MH=0.8, rinit=NULL,
           approximate_algo_delta_1=0, approximate_algo_delta_2=0, 
           nrepeats_eta=1, verbose = FALSE, totalduration = Inf, 
           xi_fixed=FALSE, sigma2_fixed=FALSE, t_dist_df){

    n <- dim(X)[1]
    p <- dim(X)[2]

    if(is.null(rinit)){
      # Initializing from the prior
      rinit <- function(){
        xi <- (1/rt(1, df=1))^2
        sigma2 <- 1/rgamma(1, shape = a0/2, rate = b0/2)
        eta <- (1/rt(p, df=t_dist_df))^2
        beta <- rnorm(p)*sqrt(sigma2/(xi*eta))
        return(list(xi = xi, sigma2 = sigma2, beta = beta, eta = eta))
      }
    }
    
    ## Initializing chains
    beta_samples1 <- matrix(nrow = (chain_length-1)%/%thin+1, ncol = p)
    beta_samples2 <- matrix(nrow = (chain_length-1)%/%thin+1, ncol = p)
    squaredists <- rep(NA, chain_length)

    # drawing initial states
    chain1 <- rinit()
    chain2 <- rinit()
    
    xi_1_current <-     chain1$xi
    sigma2_1_current <- chain1$sigma2
    beta_1_current <-   chain1$beta
    eta_1_current <-    chain1$eta
    xi_2_current <-     chain2$xi
    sigma2_2_current <- chain2$sigma2
    beta_2_current <-   chain2$beta
    eta_2_current    <- chain2$eta
    
    if(burnin == 0){ # Store initialization, and squared distance
      beta_samples1[1,]  <-  beta_1_current
      beta_samples2[1,]  <-  beta_2_current
      squaredists[1] <- sum((beta_1_current - beta_2_current)^2)
    }
    
    rm(chain1, chain2)
    #chain1 <- list('beta'=beta_1_current, 'eta'=eta_1_current, 'sigma2'=sigma2_1_current, 'xi'=xi_1_current)
    #chain2 <- list('beta'=beta_2_current, 'eta'=eta_2_current, 'sigma2'=sigma2_2_current, 'xi'=xi_2_current)

    # Setting up coupled chain
    for(iter in 1:(chain_length + burnin - 1)){
      output <-
        coupled_half_t_kernel(X, X_transpose, y, a0, b0, std_MH,
                              xi_1_current, xi_2_current, sigma2_1_current, sigma2_2_current,
                              beta_1_current, beta_2_current, eta_1_current, eta_2_current,
                              approximate_algo_delta_1 = approximate_algo_delta_1,
                              approximate_algo_delta_2 = approximate_algo_delta_2,
                              nrepeats_eta = nrepeats_eta, verbose = verbose,
                              xi_fixed=xi_fixed, sigma2_fixed=sigma2_fixed, t_dist_df)
      
      xi_1_current <- output$xi_1_samples
      sigma2_1_current <- output$sigma2_1_samples
      beta_1_current <- output$beta_1_samples
      eta_1_current <- output$eta_1_samples
      xi_2_current <- output$xi_2_samples
      sigma2_2_current <- output$sigma2_2_samples
      beta_2_current <- output$beta_2_samples
      eta_2_current <- output$eta_2_samples
      
      if(iter>=burnin) squaredists[iter-burnin+1] <- sum((beta_1_current - beta_2_current)^2)
      if((iter-burnin) %% thin == 0){
        beta_samples1[(iter-burnin)/thin+1,] <- beta_1_current
        beta_samples2[(iter-burnin)/thin+1,] <- beta_2_current
      }
    }
    
      final_output <- 
        list('beta_samples1'=beta_samples1, 
             'beta_samples2'=beta_samples2,
             "squaredist"=mean(squaredists))
    
    return(final_output)
  }



###############################################################################
#### Functions for the eta updates for blocked Gibbs sampler with half-t priors
###############################################################################
#### Single chain MCMC functions
## eta updates given beta, xi, sigma2

low_inc_gamma <- function(shape,x) low_inc_gamma_cpp(shape,x)
low_inc_gamma_inv <- function(shape,p) low_inc_gamma_inv_cpp(shape,p)

# Perfect sampling from univariate p(x) \prop x^(poly_exponent-1)*exp(-rate*x) on [0,trunc_upper]
r_trunc_poly_exp_crn <- function(poly_exponent, rate, trunc_upper, unif){
  u <- unif*(low_inc_gamma(poly_exponent, rate*trunc_upper)) # Incomplete Gamma function and inverse are from boost
  return(low_inc_gamma_inv(poly_exponent, u)/rate)
}
r_trunc_poly_exp <- function(poly_exponent, rate, trunc_upper){
  unif <- runif(length(rate))
  return(r_trunc_poly_exp_crn(poly_exponent, rate, trunc_upper, unif))
}

# Slice sampling from univariate p(x) \prop x^((v-1)/2)/(1+vx)^((v+1)/2)*exp(-mx) on [L,Inf]
eta_update_half_t <- function(xi, sigma2, beta, eta, t_dist_df, nrepeats=1)
{
  rate <- (beta^2)*(xi)/(2*sigma2)
  p <- length(eta)
  if (is.infinite(nrepeats)){
    stop('Perfect sampling not implemented for general t distribution shrinkage priors')
  } else {
    for (irepeat in 1:nrepeats){
      u <- runif(p)/(1 + t_dist_df*eta)^((1+t_dist_df)/2)
      eta <- r_trunc_poly_exp((1+t_dist_df)/2, rate,
                              (u^(-2/(1+t_dist_df))-1)/t_dist_df)
    }
  }
  return(eta)
}

###############################################################################
## CRN Coupling of Eta Update
eta_update_half_t_crn_couple <- function(xi_1, Beta_1, eta_1, sigma2_1,
                                         xi_2, Beta_2, eta_2, sigma2_2,
                                         t_dist_df){
  p <- length(eta_1)
  rate_1 <- (Beta_1^2)*(xi_1)/(2*sigma2_1)
  rate_2 <- (Beta_2^2)*(xi_2)/(2*sigma2_2)
  unif_crn_1 <- runif(p)
  u_1 <- unif_crn_1/(1 + t_dist_df*eta_1)^((1+t_dist_df)/2)
  u_2 <- unif_crn_1/(1 + t_dist_df*eta_1)^((1+t_dist_df)/2)
  unif_crn_2 <- runif(p)
  eta_1 <- r_trunc_poly_exp_crn((1+t_dist_df)/2, rate_1, (u_1^(-2/(1+t_dist_df))-1)/t_dist_df, unif_crn_2)
  eta_2 <- r_trunc_poly_exp_crn((1+t_dist_df)/2, rate_2, (u_2^(-2/(1+t_dist_df))-1)/t_dist_df, unif_crn_2)
  return(cbind(eta_1, eta_2))
}
