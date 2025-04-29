############
# 4. Plot
############
load(file = "benchmark.RData")
df_plt <- rbind(timing_assignment, timing_jackknife)

df_plt$Computation <- as.factor(df_plt$Computation)
levels(df_plt$Computation) <- c("Assigment (Bonneel et al)", 
                                "Assigment (Guthe and Thuerck)", 
                                "Cost matrix", 
                                "Flapjack algorithm")

library(ggplot2)

plt <-
  ggplot(df_plt, aes(x = n, y = walltime, 
                   color = Computation, linetype = d, fill = Computation, group = interaction(d, Computation))) +
  stat_summary(geom = "line", fun = mean) +
  stat_summary(geom = "ribbon", fun.data = mean_se, fun.args = list(mult = sqrt(R)), # Multiplied by sqrt(R) to get the standard deviation
               alpha = 0.2, color = NA) + 
  labs(x = "Sample size", y = "Wall-time (in seconds)", linetype = "Dimension") +
  scale_x_continuous(trans = "log2",
                     breaks = scales::breaks_log(8, base = 2),
                     minor_breaks = NULL) +
  scale_y_log10(breaks = scales::breaks_log(7, base = 10), minor_breaks = NULL) +
  guides(linetype = guide_legend(override.aes=list(fill=NA, lwd = 0.6)),
         color = guide_legend(override.aes=list(lwd = 0.6))) +
  theme_bw()
plt

ggsave(filename = "benchmark.pdf",
       plot = plt,
       width = 20, height = 11, units = "cm")


# What are the trends?
lm(log10(walltime) ~ log10(n), timing_assignment[timing_assignment$Computation == "Cost matrix", ])$coeff
lm(log10(walltime) ~ log10(n), timing_assignment[timing_assignment$Computation == "Assignment (Guthe and Thuerck)", ])$coeff
lm(log10(walltime) ~ log10(n), timing_assignment[timing_assignment$Computation == "Assignment (Simplex)", ])$coeff
lm(log10(walltime) ~ log10(n), timing_jackknife)$coeff
