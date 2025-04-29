source("half_t_setup.R")

# Set up parallel MCMC sampling ####
sample_coupled_half_t <- 
  function(epsilon, ncores, .burnin = burnin, .chain_length = chain_length, .thin = thin,.no_chains = no_chains, .seed = seed,
           .X = X, .X_transpose = X_transpose, .y = y, .t_dist_df = t_dist_df){
    
    set.seed(.seed)
    .rng <- RNGseq(.no_chains, .seed)
    
    cl <- parallel::makeCluster(ncores)
    doParallel::registerDoParallel(cl)
    
    out <- 
      foreach(j=1:.no_chains, rng=.rng, .packages=c("wassersteinbound")) %dopar% { 
        source("half_t_functions.R") # Import here to avoid issues on Windows 
        
        rngtools::setRNG(rng)
                
        coupled_chain <- coupled_half_t_mcmc(burnin = .burnin, chain_length = .chain_length, thin = .thin, 
                                             X = .X, X_transpose = .X_transpose, y = .y, t_dist_df = .t_dist_df,
                                             approximate_algo_delta_1=epsilon, approximate_algo_delta_2=0)
        return(list("xs" = coupled_chain$beta_samples1, 
                    "ys" = coupled_chain$beta_samples2, 
                     "squaredist" = coupled_chain$squaredist))
      }
    parallel::stopCluster(cl)
    
    return(out)
  }
#####

# Sample and save ####

# For command line usage
args <- commandArgs(trailingOnly = TRUE)
epsilon <- as.double(args[1]) # c(0.03, 0.01, 0.003, 0.001)
ncores <- as.integer(args[2])

coupled_output <- sample_coupled_half_t(epsilon, ncores)
coupled_output <- purrr::list_transpose(coupled_output)
save(coupled_output, no_chains,
     burnin, chain_length, thin,
     epsilon, file = paste0("half_t_epsilon=",epsilon,".Rdata"))

# Equivalent output obtained with:
# epsilons <- c(0.03, 0.01, 0.003, 0.001) # Approximation parameters to sweep through
# ncores <- NA # Set depending on availability
# for(epsilon in epsilons){
#   coupled_output <- sample_coupled_half_t(epsilon, ncores)
#   save(coupled_output, file = paste0("half_t_epsilon=",epsilon,".Rdata"))
# }
# 
# dim(coupled_output[[1]]$xs)
# coupled_output[[1]]$xs[,1:4]
