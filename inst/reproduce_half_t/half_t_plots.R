source('../bias_helpers.R') # Auxiliary helpers

epsilons <- c(0.03, 0.01, 0.003, 0.001, 0.0003)

ci_level <- 0.95
z_value <- qnorm(0.5 + 0.5 * ci_level)

# Function to correct bias of lower bound estimate
correct_bias <- function(estim){
  estim$w2sq <- ifelse(!is.na(estim$bias), estim$w2sq - estim$bias, estim$w2sq)
  estim$bias <- NULL
  return(estim)
}

# Read in the estimators and convert them for plotting
plot_df <- data.frame()
var_improvement_df <- data.frame()
for(epsilon in epsilons){
  load(file = paste0("half_t_epsilon=",epsilon,"_estimators.Rdata"))
  
# Get confidence intervals for plot
  # Also jackknife-correct the bias of the lower bound estimate
  plot_df <- rbind(plot_df, cbind(convert_and_get_ci(correct_bias(estimators), z_value), "Epsilon" = factor(epsilon)))
  
  # Get variance improvement
  var_improvement_df <- rbind(var_improvement_df, cbind(var_improvement, "Epsilon" = factor(epsilon)))
}

print("Estimated variance improvement factors:")
var_improvement_df


# Plot
library(ggplot2)
library(latex2exp)

# Order factors for legend
plot_df$Estimator <- factor(plot_df$Estimator, level = c("U", "L", "Coupling", "Tractable lower bound"))
levels(plot_df$Estimator)[3] <- "Coupling bound"
plot_df$Epsilon <- factor(plot_df$Epsilon, level = sort(epsilons))

plt <-
  ggplot(plot_df, aes(x = Epsilon, y = w2sq, color = Estimator)) +
  geom_point() +
  geom_errorbar(aes(ymin = ifelse(ci_lo<0, 0, ci_lo), ymax = ci_up), width = 0.4)+
  scale_y_log10() +
  #coord_cartesian(ylim=c(2e-5,NA)) +
  scale_x_discrete(labels = function(x)scales::scientific(as.numeric(x),digits=4)) +
  geom_hline(aes(yintercept=trace_of_cov,linetype="dashed"))+
  labs(y = TeX('Squared Wasserstein distance'),
       x = TeX('Approximation parameter $\\epsilon$')) +
  theme_bw(base_size = 12) +
  scale_color_discrete(name = NULL, 
                       breaks = c("U", "L", "Coupling bound", "Tractable lower bound")) +
  scale_linetype_manual(name = NULL, 
                        values = c("dashed" = "dashed"), 
                        labels = c("Trace of posterior covariance")) +
  theme(panel.grid.minor = element_blank()) +
  guides(colour = guide_legend(order = 1), 
         linetype = guide_legend(order = 2))
plt

ggsave(filename = "bias_half_t.pdf",
       plot = plt,
       width = 18, height = 9, units = "cm")

# Check how bias behaves with epsilon.
# Linear fit on log10-log10 scale, i.e. exponent and log10(constant of proportionality)
lm(log10(plot_df$w2sq[plot_df$Estimator=="Coupling bound"]) ~ log10(epsilons))
lm(log10(plot_df$w2sq[plot_df$Estimator=="U"]) ~ log10(epsilons))
lm(log10(plot_df$w2sq[plot_df$Estimator=="Tractable lower bound"]) ~ log10(epsilons))
lm(log10(plot_df$w2sq[plot_df$Estimator=="L"]) ~ log10(epsilons))

