This repository contains the R package `wassersteinbound` which reproduces the numerical experiments in 
"Bounds on Wasserstein distances between continuous distributions using independent samples" (2022)
by Tamas Papp and Chris Sherlock.


The package can be installed by using `devtools`:

```
install.packages("devtools")
devtools::install_github("tamaspapp/wassersteinbound")
```


The `/inst/` directory contains the scripts which reproduce the experiments:
1. Run all scripts in directories `./reproduceprelim`, `./reproducejack` and `./reproduceoverdispmcmc`
to replicate the results in Sections 2.4, 3.3 and 4.1.2 (and its associated appendix). 
2. Run `./reproducegibbs/gibbs_run.R` to replicate the results in Section 4.4.1. 
3. Run `./reproduceulamalascaling/_run.R` to replicate the results in Section 4.4.2 and the associated appendix.
4. Run `./reproducesvm/_run.R` to replicate the results in Section 4.4.3 and the associated appendix.
