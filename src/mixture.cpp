#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include "rng.h"

using Eigen::ArrayXd;
using Eigen::Map;
using Eigen::Ref;

using Rcpp::List;
using Rcpp::LogicalVector;
using Rcpp::Named;
using Rcpp::NumericVector;

inline double logsumexp(const Ref<const ArrayXd> &exponents)
{
    double max = exponents.maxCoeff();
    return max + log((exponents - max).exp().sum());
}

// Target, a mixture of Gaussians: pi = \sum_i p_i * N(mu_i, sigma^2)
// Use logsumexp throughout to try and avoid underflow
inline double logpi(const double &x, 
                    const Ref<const ArrayXd> &log_p, 
                    const Ref<const ArrayXd> &mu, 
                    const Ref<const ArrayXd> &inv_sigma,
                    const Ref<const ArrayXd> &log_inv_sigma)
{
    ArrayXd exponents = log_p + log_inv_sigma - 0.5 * ((x - mu) * inv_sigma).square();
    return logsumexp(exponents);
}
// Target gradient
inline double gradlogpi(const double &x, 
                        const Ref<const ArrayXd> &log_p, 
                        const Ref<const ArrayXd> &mu, 
                        const Ref<const ArrayXd> &inv_sigma,
                        const Ref<const ArrayXd> &log_inv_sigma)
{
    ArrayXd exponents_numerator   = log_p + 3 * log_inv_sigma - 0.5 * ((x - mu) * inv_sigma).square();
    double max_numerator = exponents_numerator.maxCoeff();
    ArrayXd exponents_denominator = log_p + log_inv_sigma - 0.5 * ((x - mu) * inv_sigma).square();

    return - exp(max_numerator) * ((x - mu) * (exponents_numerator - max_numerator).exp()).sum() / exp(logsumexp(exponents_denominator));
}

// "x0" = starting point
// "h" = step size
// "iter" = maximum number of iterations
//
// p = mixture probs
// mu = mixture means
// sigma = mixture standard deviations
//
//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List RWMMixture1d(const double &x0, 
                        const Eigen::Map<Eigen::ArrayXd> &p, 
                        const Eigen::Map<Eigen::ArrayXd> &mu, 
                        const Eigen::Map<Eigen::ArrayXd> &sigma,
                        const double &h, 
                        const int &iter)
{
    ArrayXd inv_sigma = 1 / sigma;
    ArrayXd log_inv_sigma = inv_sigma.log();
    ArrayXd log_p = p.log();
    
    // Declare temporary variables
    double x = x0, xp;
    double logpi_x = logpi(x, log_p, mu, inv_sigma, log_inv_sigma), logpi_xp, logHR_x, u;

    // Intialize storage
    LogicalVector acc(iter);
    NumericVector xs(iter + 1);
    xs(0) = x;

    // "iter" iterations of MCMC
    for (int i = 1; i <= iter; i++)
    {
        // Generate proposal
        xp = x + h * RNG::rnorm(RNG::rng);
        logpi_xp = logpi(xp, log_p, mu, inv_sigma, log_inv_sigma);

        // Compute log of Hastings ratio
        logHR_x = logpi_xp - logpi_x;

        // Perform acceptance step
        u = RNG::runif(RNG::rng);
        if (logHR_x > 0 || log(u) < logHR_x)
        {
            acc(i - 1) = true;
            x = xp;
            logpi_x = logpi_xp;
        } 
        else
        {
            acc(i - 1) = false;
        }
        xs(i) = x;
    }

    return Rcpp::List::create(Rcpp::Named("xs")  = xs,
                              Rcpp::Named("acc") = acc);
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List MALAMixture1d(const double &x0, 
                         const Eigen::Map<Eigen::ArrayXd> &p, 
                         const Eigen::Map<Eigen::ArrayXd> &mu, 
                         const Eigen::Map<Eigen::ArrayXd> &sigma,
                         const double &h, 
                         const int &iter)
{
    // Constants
    ArrayXd inv_sigma = 1 / sigma;
    ArrayXd log_inv_sigma = inv_sigma.log();
    ArrayXd log_p = p.log();
    double h_sq = h * h;
    double invh_sq = 1 / h_sq;

    // Declare temporary variables
    double x = x0, xp, z;
    double x_mean = x + 0.5 * h_sq * gradlogpi(x, log_p, mu, inv_sigma, log_inv_sigma), xp_mean;
    double logpi_x = logpi(x, log_p, mu, inv_sigma, log_inv_sigma), logpi_xp, logq_x_xp, logq_xp_x, logHR_x, u;

    // Intialize storage
    LogicalVector acc(iter);
    NumericVector xs(iter + 1);
    xs(0) = x;
    
    // "iter" iterations of MCMC
    for (int i = 1; i <= iter; i++)
    {
        // Generate proposal
        z = RNG::rnorm(RNG::rng);
        xp = x_mean + h * z;

        logpi_xp = logpi(xp, log_p, mu, inv_sigma, log_inv_sigma);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, log_p, mu, inv_sigma, log_inv_sigma);

        // Proposal log-densities
        logq_xp_x = -0.5 * invh_sq * (x - xp_mean) * (x - xp_mean); // Go from xp to x
        logq_x_xp = -0.5 * z * z;                                   // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Perform acceptance step
        u = RNG::runif(RNG::rng);
        if (logHR_x > 0 || log(u) < logHR_x)
        {
            acc(i - 1) = true;
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        } 
        else
        {
            acc(i - 1) = false;
        }
        xs(i) = x;
    }

    return Rcpp::List::create(Rcpp::Named("xs")  = xs,
                              Rcpp::Named("acc") = acc);
}
