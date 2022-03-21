
library(wassersteinbound)
library(Matrix)

D    <- rev(as.integer(c(50, 100, 200, 300, 400, 500, 600, 800, 1000)))
Lo   <- c(1570, 1300, 1020, 880, 720, 500, 300, 200, 50)
High <- c(1575, 1320, 1040, 900, 740, 600, 500, 300, 200)

disp <- sqrt(3)

w2sq_thresh <- 6

w2sq_mix_true <- rep(NA, length(D))
                     
in_time <- Sys.time()

for(i in 1:length(D)) {
  
  d <- D[i]
  lo <- Lo[i]
  high <- High[i]
  
  mu_0 <- mu <- rep(0, d)
  Sigma_0 <- as(disp^2 * diag(d), "dgCMatrix")
  
  h <- 0.2 / d^(1/4)
  
  target_mats <- GenerateSigmaAndU_scaling(d)
  Sigma_inv <- as(target_mats$Sigma_inv, "dgCMatrix")
  
  Sigma_ULA_inv <- as((diag(d) - h^2 / 4 * Sigma_inv) %*% Sigma_inv, "dgCMatrix")
  Sigma_ULA <- as.matrix(solve(Sigma_ULA_inv))
  M <- diag(d) - h^2 / 2 * Sigma_inv
  
  which_iter <- as.integer(seq(from = lo, to = high, by = 1))
  
  out <- EvaluateW2sqULA(mu_0, Sigma_0, mu, Sigma_ULA, M, which_iter, h)
  
  w2sq <- out$w2sq
  print(w2sq)
  
  w2sq_mix_true[i] <- which_iter[match(TRUE, w2sq < w2sq_thresh)]
}

out_time <- Sys.time()

out_time - in_time

plot(log(D, 10), log(w2sq_mix_true, 10))

lin <- lm(log(w2sq_mix_true, 10) ~ log(D, 10))
summary(lin)
abline(lin)

save(w2sq_mix_true, file = "ula_exact_mixing.Rdata")

