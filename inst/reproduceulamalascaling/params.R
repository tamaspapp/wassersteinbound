R <- 1000L

seed <- 12345L

ncores <- parallel::detectCores()

D <- rev(as.integer(c(50, 100, 200, 300, 400, 500, 600, 800, 1000)))