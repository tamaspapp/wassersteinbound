// Samplers for standard Gaussian target

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include "rng.h"

using Eigen::Map;
using Eigen::Ref;
using Eigen::VectorXd;

using Rcpp::List;
using Rcpp::LogicalVector;
using Rcpp::Named;
using Rcpp::NumericVector;

// Target: pi = N(0, I)
inline double logpi(const Ref<const VectorXd> &x)
{
    return -0.5 * x.squaredNorm();
}
inline VectorXd gradlogpi(const Ref<const VectorXd> &x)
{
    return -x;
}

// "x0" = starting point
// "h" = step size
// "iter" = number of iterations
//
//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List RWMStdGaussian(const Eigen::Map<Eigen::VectorXd> &x0, 
                          const double &h, 
                          const int &iter)
{   
    // Constants
    int d = x0.rows();

    // Declare temporary variables
    VectorXd x = x0, xp(d);
    double logpi_x = logpi(x), logpi_xp, logHR_x, u;

    // Intialize storage
    LogicalVector acc(iter);
    NumericVector x_norms(iter + 1);
    x_norms(0) = x.norm();

    // "iter" iterations of MCMC
    for (int i = 1; i <= iter; i++)
    {
        // Generate proposal
        for (int j = 0; j < d; j++)
        {
            xp(j) = RNG::rnorm(RNG::rng);
        }
        xp = x + h * xp;
        logpi_xp = logpi(xp);

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
        x_norms(i) = x.norm();
    }

    return Rcpp::List::create(Rcpp::Named("x_norms") = x_norms,
                              Rcpp::Named("acc")     = acc);
}                 

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List MALAStdGaussian(const Eigen::Map<Eigen::VectorXd> &x0, 
                           const double &h, 
                           const int &iter)
{
    // Constants
    double h_sq = h * h;
    double invh_sq = 1 / h_sq;
    int d = x0.rows();

    // Declare temporary variables
    VectorXd x = x0;
    VectorXd xp(d), z(d);
    VectorXd x_mean = x + 0.5 * h_sq * gradlogpi(x), xp_mean(d);
    double logpi_x = logpi(x), logpi_xp, logq_x_xp, logq_xp_x, logHR_x, u;

    // Intialize storage
    LogicalVector acc(iter);
    NumericVector x_norms(iter + 1);
    x_norms(0) = x.norm();
    
    // "iter" iterations of MCMC
    for (int i = 1; i <= iter; i++)
    {
        // Generate proposal
        for (int j = 0; j < d; j++)
        {
            z(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * z;

        logpi_xp = logpi(xp);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp);

        // Proposal log-densities
        logq_xp_x = -0.5 * invh_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
        logq_x_xp = -0.5 * z.squaredNorm();                       // Go from x to xp

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
        x_norms(i) = x.norm();
    }

    return Rcpp::List::create(Rcpp::Named("x_norms") = x_norms,
                              Rcpp::Named("acc")     = acc);
}
