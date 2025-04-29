
if(!file.exists("convergence_ar1_unadjusted.RData")){source("convergence_ar1_unadjusted.R")}
load("convergence_ar1_unadjusted.RData")
if(!file.exists("convergence_ar1_adjusted.RData")){source("convergence_ar1_adjusted.R")}
load("convergence_ar1_adjusted.RData")

library(ggplot2)

mixing_times_unadjusted$algorithm <- ifelse(mixing_times_unadjusted$gamma == "Overdamped", "(a) ULA", "(c) OBABO")
mixing_times_adjusted$algorithm <- ifelse(mixing_times_adjusted$gamma == "Overdamped", "(b) MALA", "(d) Horowitz")

mixing_times <- rbind(mixing_times_unadjusted, mixing_times_adjusted)


### Manual plot fiddling
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
colors <- c(gg_color_hue(3), "black")
mixing_times$Estimator[mixing_times$Estimator == "Coupling"] <- "Coupling bound"
mixing_times$Estimator[mixing_times$Estimator == "Exact"] <- "Exact mixing time"
levels <- c("U", "L", "Coupling bound", "Exact mixing time")
###


plt <-
  ggplot(mixing_times, aes(x = d, y = iter, color = Estimator)) +
  geom_line() + geom_point() +
  facet_grid(~algorithm) +
  scale_y_log10(breaks = scales::breaks_log(7,2)) + scale_x_log10(breaks = scales::breaks_log(5,2)) +
  theme_bw() +
  labs(x = "Dimension", y = "Mixing time") +
  theme(panel.grid.minor = element_blank(),
        panel.spacing = unit(0.75, "lines"),
        strip.text.x = element_text(hjust = 0, margin=margin(l=0,b=4), size = 11),
        strip.background = element_blank())+#,
        #legend.position = "bottom") +
  scale_color_manual(name=NULL, values = colors, breaks = levels) 
plt

ggsave(filename = "mixing_ar1.pdf",
       plot = plt,
       width = 24, height = 8, units = "cm")
