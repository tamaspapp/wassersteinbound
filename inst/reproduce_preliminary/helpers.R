squaredSign <- function(x) sign(x) * x^2

get_true_var <- function(out){
  df <- melt(out, id.vars = c("d", "n", "sigma_sq"),
             measure.vars = c("Unbiased"),
             value.name = "w2sq", variable.name = "Estimator")
  
  true_var <- df %>%
    group_by(d, n, sigma_sq, Estimator) %>%
    summarise(Variance = var(w2sq))
  
  return(true_var)
}
get_true_var_boot_ci <- function(true_out, true_reps, boot_reps = 5e2, coverage = 0.95) {
  
  # Bootstrap subsample for each group
  groupwise_subsample <- function(idx){
    l <- length(idx)
    num_groups <- nrow(true_out) %/% l 
    idx_rep <- matrix(seq(1, l * num_groups), nrow = l)
    idx_rep <- idx_rep[idx, ]
    return(as.vector(idx_rep))
  }
  
  dfs <- vector("list", boot_reps)
  for(i in 1:boot_reps){
    # Shuffle the indices in all groups
    idx <- sample.int(true_reps,replace = T)
    dfs[[i]] <- true_out[groupwise_subsample(idx),]
    dfs[[i]]$"boot_rep" <- i # Store with bootstrap replicate index
  }
  dfs <- do.call(rbind, dfs)
  
  dfs <- melt(dfs, id.vars = c("d", "n", "sigma_sq","boot_rep"),
              measure.vars = c("Unbiased"),
              value.name = "w2sq", variable.name = "Estimator")
  
  # Variance for each bootstrap replicate  
  var_boot_df <- dfs %>%  
    group_by(d, n, sigma_sq, Estimator, boot_rep) %>%
    summarise(Variance = var(w2sq))
  
  # Boostrap confidence interval for the variance
  var_boot_cis <- var_boot_df %>%  
    group_by(d, n, sigma_sq, Estimator) %>%
    summarise(ci_up = quantile(Variance, (1+coverage)/2), ci_lo = quantile(Variance, (1- coverage)/2))
  return(var_boot_cis)
}


sigma_sq_label_fun <- function(string) TeX(paste0("$\\sigma^2 = $",string))
sigma_sq_label <- as_labeller(sigma_sq_label_fun, default = label_parsed)
d_label_fun <- function(string) TeX(paste0("$d = $",string))
d_label <- as_labeller(d_label_fun, default = label_parsed)
n_label_fun <- function(string) TeX(paste0("$n = $",string))
n_label <- as_labeller(n_label_fun, default = label_parsed)

# get_box_plots <- function(estimated, unbiased){
# 
#   ggplot(estimated) +
#     facet_nested(d~sigma_sq+n, scales = "free", independent = "y",
#                  labeller = labeller(sigma_sq = sigma_sq_label, d = d_label, n = n_label)) +
#     geom_point(unbiased, mapping = aes(x = Estimator, y = Variance, color = Estimator)) +
#     geom_errorbar(unbiased, mapping = aes(x = Estimator, ymax = ci_up, ymin = ci_lo, color = Estimator)) +
#     geom_boxplot(mapping = aes(x = Estimator, y = Variance, fill = Estimator, group = interaction(n, Estimator))) +
#     theme_bw() +
#     scale_y_continuous(minor_breaks = NULL) +
#     scale_x_discrete(limits=c(levels(estimated$Estimator),"Unbiased"), breaks = NULL) +
#     guides(fill = guide_legend(override.aes = list(shape = NA), order = 1),
#            color = guide_legend(order = 2)) +
#     labs(x = NULL, color = NULL)
# }
