library(ggplot2)
source("../helpers.R")

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

load(file = "rwm_plot.RData")
load(file = "ula_plot.RData")
load(file = "mala_plot.RData")
load(file = "fmala_plot.RData")

general_plot_params <- list(
  geom_hline(yintercept = 0, linetype = "dashed"),
  geom_line(),
  geom_ribbon(aes(ymin = ci_up, ymax = ci_lo), alpha = 0.15, colour = NA),
  coord_cartesian(ylim = c(-0.5, 500)),
  scale_y_continuous(trans = log.const.p(2),
                     breaks = c(0,1,3,10,30,100,300, 1000), labels = scales::label_number(),
                     minor_breaks = NULL),
  labs(y = "Squared Wasserstein distance", color = "Estimator"),
  theme_bw(),
  guides(colour = guide_legend(override.aes = list(lwd=0.75))),
  theme(strip.text.x = element_text(hjust = 0, margin=margin(l=0,b=4), size = 11),
        strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom"),
)


# Joint plot for appendix
  # RWM: experiments with GCRN coupling
  # ULA: sanity check on Laplace approximation
appendix_df <- 
  rbind(
    cbind(rwm_big_gcrn[rwm_big$iter < 1e6, ], "alg" = "(a) RWM: optimal step size"),
    cbind(rwm_small_gcrn[rwm_small$iter < 1e6, ], "alg" = "(b) RWM: small step size"),
    cbind(plt_df_ula[plt_df_mala$iter < 1e4, ], "alg" = "(c) ULA on Laplace approximation")
  )

## Manual plot fiddling
appendix_df$estimator[appendix_df$estimator == "Coupling"] <- "Coupling bound"
appendix_df$estimator[appendix_df$estimator == "Exact"] <- "Exact squared Wasserstein"

#appendix_df$estimator <- factor(appendix_df$estimator, levels = levels)
#levels(appendix_df$estimator)[c(3,4)] <- c("Coupling bound", "Exact squared Wasserstein")

levels <- c("U", "L", "Coupling bound", "Exact squared Wasserstein")

colors <- c(gg_color_hue(3), "black")
fills <- c(gg_color_hue(3), NA)
##

appx_plt <-
  ggplot(appendix_df,
         aes(x = iter / 1000, y = w2sq, color = estimator, fill = estimator)) +
  facet_grid(~alg, scales = "free_x") +
  general_plot_params + 
  scale_color_manual(name = NULL, values = colors, breaks = levels) +
  scale_fill_manual(name = NULL, values = fills, na.value="transparent", breaks = levels) +
  labs(x = "Iteration (x1000)")
appx_plt


ggsave(filename = "svm_appendix.pdf",
       plot = appx_plt,
       width = 24, height = 9, units = "cm")


# Joint plot for main text
joint_df <- 
  rbind(
    cbind(rwm_big[rwm_big$iter < 1e6, ], "alg" = "(a) RWM: optimal step size"),
    cbind(rwm_small[rwm_small$iter < 1e6, ], "alg" = "(b) RWM: small step size"),
    cbind(plt_df_mala[plt_df_mala$iter < 1e4, ], "alg" = "(c) MALA"),
    cbind(plt_df_fmala[plt_df_fmala$iter < 1e4, ], "alg" = "(d) Fisher-MALA")
  )

joint_df$estimator[joint_df$estimator == "L_sq"] <- "L"

## Manual plot fiddling
joint_df$estimator[joint_df$estimator == "Coupling"] <- "Coupling bound"

#appendix_df$estimator <- factor(appendix_df$estimator, levels = levels)
#levels(appendix_df$estimator)[c(3,4)] <- c("Coupling bound", "Exact squared Wasserstein")

levels <- c("U", "L", "Coupling bound")

colors <- gg_color_hue(3)
fills <- gg_color_hue(3)
##


scientific_10 <- function(x) parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))

joint_plt <-
  ggplot(joint_df, 
         aes(x = iter / 1000, y = w2sq, color = estimator, fill = estimator)) +
  general_plot_params +
  facet_wrap(~alg, nrow = 1, scales = "free_x") +
  scale_color_manual(name = NULL, values = colors, breaks = levels) +
  scale_fill_manual(name = NULL, values = fills, na.value="transparent", breaks = levels) +
  labs(x = "Iteration (x1000)")
joint_plt

ggsave(filename = "svm_main.pdf",
       plot = joint_plt,
       width = 24, height = 9, units = "cm")

