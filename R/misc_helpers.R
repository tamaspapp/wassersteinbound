
#' One-dimensional quadratic transportation cost via order statistics
#'
#' x, y = vector of empirical measures, must be same length
#' 
#' order_x, order_y = vectors of indices, denote increasing order for data 
#' x and y
#'
#' @export
ComputeTranspCost1d <- function(x, y, order_x, order_y) {
  return(sum((x[order_x] - y[order_y])^2))
}

#' "n-out-of-n" boostrap variance estimate from R repeats
#' 
#' x = vector, data to be resampled
#' 
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
