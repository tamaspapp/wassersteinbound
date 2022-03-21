# Upper bound, w2sq
w2sq_ub          <- w2sq - mean(w2sq[debias_iters])

# Upper bound, w2
w2_ub          <- sign(w2sq_ub) * sqrt(abs(w2sq_ub))

# Lower bound, w2 
w2      <- sqrt(w2sq)

w2_lb   <- w2 - mean(w2[debias_iters])

# Lower bound, w2sq
w2sq_lb          <- sign(w2_lb) * w2_lb^2