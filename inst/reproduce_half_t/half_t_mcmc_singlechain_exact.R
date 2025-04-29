source("half_t_setup.R")

# Set up parallel MCMC sampling ####
sample_singlechain_half_t <- 
  function(ncores, .burnin = burnin, .chain_length = chain_length, .thin = thin,.no_chains = no_chains, .seed = seed,
           .X = X, .X_transpose = X_transpose, .y = y, .t_dist_df = t_dist_df){
    
    set.seed(.seed)
    
    .rng <- tail(RNGseq(2 * .no_chains, .seed), .no_chains)# Second half of the rng streams for the additional exact samples
    
    cl <- parallel::makeCluster(ncores)
    doParallel::registerDoParallel(cl)
    
    out <- 
      foreach(j=1:.no_chains, rng=.rng, .packages=c("wassersteinbound")) %dopar% { 
        source("half_t_functions.R") # Import here to avoid issues on Windows 
                  
        rngtools::setRNG(rng)
        
        single_chain <- half_t_mcmc(burnin=.burnin, chain_length=.chain_length, thin=.thin, 
                                    X=.X, X_transpose=.X_transpose, y=.y, t_dist_df=.t_dist_df,
                                    approximate_algo_delta=0, store_all=F)

        return(list("ys" = single_chain$beta_samples))
      }
    parallel::stopCluster(cl)
    
    return(out)
  }
#####

args <- commandArgs(trailingOnly = TRUE)
ncores <- as.integer(args[1])

# Sample and save ####
singlechain_output <- sample_singlechain_half_t(ncores)
singlechain_output <- purrr::list_transpose(singlechain_output)
save(singlechain_output, file = paste0("half_t_singlechain_exact.Rdata"))
