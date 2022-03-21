
GetJackknifeVariance <- function(jack_estimates, R) {
  return((R - 1) / R * (apply(jack_estimates, 2, var) * (R - 1)))
}


# Upper bound, w2sq
w2sq_ub          <- w2sq - mean(w2sq[debias_iters])
jack_w2sq_ub     <- jack_w2sq - rowMeans(jack_w2sq[, debias_iters])
w2sq_ub_jack_var <- GetJackknifeVariance(jack_w2sq_ub, R)

# Upper bound, w2
w2_ub          <- sign(w2sq_ub) * sqrt(abs(w2sq_ub))
jack_w2_ub     <- sign(jack_w2sq_ub) * sqrt(abs(jack_w2sq_ub))
w2_ub_jack_var <- GetJackknifeVariance(jack_w2_ub, R)

# Lower bound, w2
w2      <- sqrt(w2sq)
jack_w2 <- sqrt(jack_w2sq)

w2_lb          <- w2 - mean(w2[debias_iters])
jack_w2_lb     <- jack_w2 - rowMeans(jack_w2[, debias_iters])
w2_lb_jack_var <- GetJackknifeVariance(jack_w2_lb, R)

# Lower bound, w2sq
w2sq_lb          <- sign(w2_lb) * w2_lb^2
jack_w2sq_lb     <- sign(jack_w2_lb) * jack_w2_lb^2
w2sq_lb_jack_var <- GetJackknifeVariance(jack_w2sq_lb, R)