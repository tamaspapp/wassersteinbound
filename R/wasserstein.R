
#####
# Squared Wasserstein distance estimation
#####

#' Calculate the squared empirical Wasserstein distance, with the quadratic cost, 
#' by solving a linear assignment problem.
#'
#' @export
w2sq_empirical <- function(x, y, estimate_epsilon = TRUE){
  
  n <- nrow(x)
  if(nrow(y) != n) stop("Only balanced sample sizes are supported.")
  
  c_out <- wassersteinbound::assignment_squared_euclidean_toms1015_cpp(x,y,estimate_epsilon)
  
  return(list("w2sq"                      = c_out$cost / n,
              "assignment_cost_fractions" = c_out$cost_fractions,     
              "potentials_x"              = c_out$row_potentials,
              "potentials_y"              = c_out$column_potentials))  
}


#' Calculate the squared empirical Wasserstein distance, with the quadratic cost, 
#' by solving a linear assignment problem. Then, compute the jackknife
#' (i.e. leave-one-pair-of-samples-out) assignment costs.
#' 
#' @export
w2sq_empirical_jack <- function(x, y, estimate_epsilon = TRUE){
  
  n <- nrow(x)
  if(nrow(y) != n){
    stop("Only balanced sample sizes are supported.")
  }
  
  c_out <- wassersteinbound::assignment_squared_euclidean_jackknife_cpp(x,y,estimate_epsilon)
  
  return(list("w2sq"                       = c_out$cost / n,
              "assignment_cost_fractions"  = c_out$cost_fractions, 
              "potentials_x"               = c_out$row_potentials,
              "potentials_y"               = c_out$column_potentials,
              "jackknife_assignment_costs" = c_out$jack_costs)) 
}


#####
# One-dimensional case
#####

#' One-dimensional quadratic transportation cost via order statistics
#'  and potentials via an O(n) post-processing step (Algorithm 3 of https://proceedings.mlr.press/v151/sejourne22a/sejourne22a.pdf)
#'
#' @export
w2sq_empirical_1d <- function(x, y){
  
  # Helper function
  invert_order <- function(ordered, order){
    unordered <- rep(NA, length(order))
    unordered[order] <- ordered
    return(unordered)
  }
  
  n <- length(x)
  if(length(y) != n) stop("Only balanced sample sizes are supported.")
  
  order_x <- order(x)
  order_y <- order(y)
  
  out <- w2sq_1d_dual_cpp(x[order_x], y[order_y])
  
  return(list("w2sq"                      = out$w2sq, # i.e. mean((x[order_x] - y[order_y])^2)
              "assignment_cost_fractions" = out$potentials_x + out$potentials_y,
              "potentials_x"              = invert_order(out$potentials_x, order_x),
              "potentials_y"              = invert_order(out$potentials_y, order_y)))
}
