
library(wassersteinbound)
library(doParallel)
library(Matrix)

SquareSign <- function(x) { return(sign(x) * x^2)}

D <- as.integer(c(50, 100, 200, 300, 400, 500, 600, 800, 1000))
R <- 1000
reps <- 50

set.seed(12345)

w2sq <- rep(0, length(D))
w2sq_ub <- w2sq_lb <- c()

cl <- parallel::makeCluster(4)
doParallel::registerDoParallel(cl)

in_time <- Sys.time()
for(i in 1:length(D)) {
  d <- D[i]
  
  h <- 1 / d^{1/6}
  
  target_mats <- GenerateSigmaAndU_scaling(d)
  Sigma_inv <- as(target_mats$Sigma_inv, "dgCMatrix")
  
  Sigma_ULA_inv <- as((diag(d) - h^2 / 4 * Sigma_inv) %*% Sigma_inv, "dgCMatrix")
  Sigma_ULA <- as.matrix(solve(Sigma_ULA_inv))
  
  mu <- rep(0, d)
  
  Sigma <- matrix(NA, d, d)
  
  for(j in 1:d) {
    for(k in 1:d) {
      Sigma[j, k] <- 0.5 ^ abs(j - k)
    }
  }
  
  w2sq[i] <- EvaluateW2sq(mu, Sigma_ULA, mu, Sigma)
  
  A <- as(t(chol(Sigma_ULA)), "dgCMatrix")
  B <- as(t(chol(Sigma)), "dgCMatrix")
  
  out <- foreach(j = 1:reps, .packages = c("wassersteinbound", "Matrix"), .combine = c) %dopar% {
    
    x <- y <- z <- matrix(NA, nrow = d, ncol = R) 
    
    for (t in 1:R) {
      x[, t] <- as.vector(A %*% rnorm(d))
      y[, t] <- as.vector(B %*% rnorm(d))
      z[, t] <- as.vector(B %*% rnorm(d))
    }
    
    first_term  <- SolveAssignmentNetworkflow(EvaluateSquaredCost(x, z))
    second_term <- SolveAssignmentNetworkflow(EvaluateSquaredCost(y, z))
    
    w2sq_ub_ <- first_term - second_term
    w2sq_lb_ <- SquareSign(sqrt(first_term) - sqrt(second_term))
    
    a <- c(w2sq_ub_, w2sq_lb_)
    
    return(a)
  }
  
  w2sq_ub <- c(w2sq_ub, out[c(TRUE, FALSE)]) 
  w2sq_lb <- c(w2sq_lb, out[c(FALSE, TRUE)]) 
  
}
out_time <- Sys.time()
print(out_time - in_time)

parallel::stopCluster(cl)

######
# Convert data to long format

data_frame_dim   <- c(rep(sort(rep(D, reps)),2), D)
data_frame_type  <- c(rep(rep("w2sq_ub", reps), length(D)), rep(rep("w2sq_lb", reps), length(D)), rep("w2sq_exact", length(D)))
data_frame_bound <- c(w2sq_ub, w2sq_lb, w2sq)

data_frame <- data.frame(data_frame_type, data_frame_bound, data_frame_dim)
names(data_frame) <- c("type", "bound", "dimension")

save(data_frame, file = "ula_target_bias.RData")

######
# Plot

load(file = "ula_target_bias.RData")

library(ggplot2)

palette <- c("w2sq_ub" = "#56B4E9",
             "w2sq_lb" = "#E69F00",
             "w2sq_exact" = "#000000")
brks <-  c("w2sq_ub",
           "w2sq_lb",
           "w2sq_exact")
labels <- c("Upper (empirical)", 
          "Lower (empirical)", 
          "Exact")

df1 <- data_frame[data_frame$type != "w2sq_exact",]
df2 <- data_frame[data_frame$type == "w2sq_exact",]

plt <- ggplot(data = df1,aes(x = factor(dimension), y = bound)) +
  geom_boxplot(aes(fill = factor(type))) +
  geom_point(data = df2, 
             aes(color = factor(type)),
             shape = 4, size = 3) +
  coord_trans(y = "log10") +
  scale_y_continuous(breaks = c(0.03, 0.1, 0.3, 1, 3, 10, 30), minor_breaks = c()) +
  ylab("Squared Waserstein distance") +
  xlab("Dimension") +
  scale_colour_manual(values = palette,
                      breaks = brks,
                      labels = labels,
                      name = c("Bound type")) +
  scale_fill_manual(values = palette,
                    breaks = brks,
                    labels = labels,
                    name = c("Bound type")) +
  labs(fill = "Bound type", colour = "Bound type") +
  # guides(fill = guide_legend(override.aes = list(linetype = c(1, 1, 0) ) ) ) +
  # guides(fill=guide_legend(override.aes=list(fill=NA))) +
  theme_bw()
plt <- plt + theme(legend.position = "none")

ggsave(filename = "ula_target_bias.pdf", 
       plot = plt,
       width = 16, height = 12, units = "cm")

