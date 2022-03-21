#####
# Format data for convenience

tau <- rep(NA, R)

for (i in 1:R) {
  tau[i] <- out[[i]]$tau
}

x <- vector(mode = "list", length = floor(iter_store_x / thin) + 1)


a <- matrix(NA, nrow = t, ncol = R)

for (i in 1:length(x)) {
  
  for(j in 1:R) {
    a[, j] <- out[[j]]$x[[i]]
  }
  
  x[[i]] <- a
}

x_reference <- matrix(NA, nrow = t, ncol = R)

for (j in 1:R) {
  x_reference[, j] <- out[[j]]$x_final
}

w2_bound_components <- vector(mode = "list", length = R)

for (i in 1:R) {
  w2_bound_components[[i]] <- out[[i]]$squaredist
}

rm(a, out)