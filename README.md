Implements the methods and reproduces the experiments in the paper ["Centered plug-in estimation of Wasserstein distances"](https://arxiv.org/abs/2203.11627) by Tamas Papp and Chris Sherlock.

The package can be installed by using `devtools`:

```
devtools::install_github("tamaspapp/wassersteinbound")
```

The `/inst/` directory contains all the scripts to reproduce the experiments:
1. See `./reproduce_preliminary` for Sections 2 and 3.
2. See `./reproduce_synthetic_mcmc` for Sections 4.4.1 and 5.4.1.
3. See `./reproduce_talldata` for Section 4.4.2.
4. See `./reproduce_halft` for Section 4.4.3.
5. See `./reproduce_svm` for Section 5.4.2.
