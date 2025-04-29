/*
Gibbs samplers for the periodic AR(1) process of D. Wilkison, in the discussion of Jacob et al. - Unbiased MCMC with couplings (JRSSB, 2021)
*/
// The samplers only work for d >= 2. 

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <tuple>

#include "rng_pcg.h"

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using std::pair;

/*
    // Precomputed terms:
    // for conditional means
    ArrayXd q_inv = Q.diagonal().array().inverse(); // Vector of inverses of diagonal entries of Q
    ArrayXd a = q_inv * (Q * mu).array();
    MatrixXd A = MatrixXd::Identity(d, d) - q_inv.matrix().asDiagonal() * Q;

    // for conditional standard deviations
    ArrayXd sd = q_inv.sqrt();
*/

// Clip index to grid 0...(d-1)
inline int clip(int idx, const int &d)
{
    idx = (idx < 0) ? (idx + d) : idx;
    idx = (idx >= d) ? (idx - d) : idx;
    return idx;
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List gibbs_periodicAR1_cpp(const Eigen::Map<Eigen::ArrayXd> &x0, 
                                 const double &sd,
                                 const double &c, 
                                 const int &iter,
                                 const int &thin)
{
    int d = x0.size();
    ArrayXd x = x0;

    // Storage
    ArrayXXd xs(iter / thin + 1, d);
    xs.row(0) = x;

    // Gibbs steps
    for (int it = 1; it <= iter; it++)
    {
        for (int j = 0; j < d; j++)
        {
            int j_minus_1 = clip(j - 1, d);
            int j_plus_1  = clip(j + 1, d);
            double mu_x = c * (x(j_minus_1) + x(j_plus_1));

            double z = sd * RNG::rnorm(RNG::rng);
            x(j) = mu_x + z;
        }

        // Storage, thinned
        if (it % thin == 0) xs.row(it / thin) = x;
    }

    return Rcpp::List::create(Rcpp::Named("xs") = Rcpp::wrap(xs));
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List gibbs_periodicAR1_CRN_cpp(const Eigen::Map<Eigen::ArrayXd> &x0, 
                                     const Eigen::Map<Eigen::ArrayXd> &y0,
                                     const double &sd,
                                     const double &c,
                                     const int &iter,
                                     const int &thin)
{
    // Constants
    int d = x0.size();

    // Initial states
    ArrayXd x = x0;
    ArrayXd y = y0;

    // Storage, with thinning
    ArrayXXd xs(iter / thin + 1, d), ys(iter / thin + 1, d);
    ArrayXd squaredist = ArrayXd::Zero(iter/thin + 1);

    xs.row(0) = x;
    ys.row(0) = y;
    squaredist(0) = (x - y).square().sum();

    for (int it = 1; it <= iter; it++)
    {
        for (int j = 0; j < d; j++) 
        {
            int j_minus_1 = clip(j - 1, d);
            int j_plus_1  = clip(j + 1, d);

            double mu_x = c * (x(j_minus_1) + x(j_plus_1));
            double mu_y = c * (y(j_minus_1) + y(j_plus_1));
            
            double z = sd * RNG::rnorm(RNG::rng);
            x(j) = mu_x + z;
            y(j) = mu_y + z;
        }

        if (it % thin == 0) 
        {
            xs.row(it / thin) = x;
            ys.row(it / thin) = y;
            squaredist(it / thin) = (x - y).square().sum();
        }
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("squaredist") = squaredist);
}

// Univariate reflection-maximal coupling for Gaussians with means (mu_x, mu_y) and common standard deviation sd.
inline std::pair<double, double> reflmax_gaussian(const double &mu_x, const double &mu_y, const double &sd)
{
    double z = sd * RNG::rnorm(RNG::rng);
    double x = mu_x + z;
    double y;

    double log_cpl = 0.5 * (z * z - (x - mu_y) * (x - mu_y)) / (sd * sd);
    if (log_cpl > 0 || log(RNG::runif(RNG::rng)) < log_cpl)
        y = x;
    else
        y = mu_y - z;

    return std::make_pair(x, y);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List gibbs_periodicAR1_ReflMax_cpp(const Eigen::Map<Eigen::ArrayXd> &x0, 
                                         const Eigen::Map<Eigen::ArrayXd> &y0,
                                         const double &sd,
                                         const double &c,
                                         const int &iter,
                                         const int &thin)
{
    // Constants
    int d = x0.size();

    // Initial states
    ArrayXd x = x0;
    ArrayXd y = y0;

    // Storage, with thinning
    int tau = -1;
    ArrayXXd xs(iter / thin + 1, d), ys(iter / thin + 1, d);
    ArrayXd squaredist = ArrayXd::Zero(iter/thin + 1);

    xs.row(0) = x;
    ys.row(0) = y;
    squaredist(0) = (x - y).square().sum();

    for (int it = 1; it <= iter; it++)
    {
        for (int j = 0; j < d; j++)
        {
            int j_minus_1 = clip(j - 1, d);
            int j_plus_1  = clip(j + 1, d);

            double mu_x = c * (x(j_minus_1) + x(j_plus_1));
            double mu_y = c * (y(j_minus_1) + y(j_plus_1));

            auto [x_, y_] = reflmax_gaussian(mu_x, mu_y, sd);

            x(j) = x_;
            y(j) = y_; 
        }

        if (it % thin == 0) 
        {
            xs.row(it / thin) = x;
            ys.row(it / thin) = y;
            squaredist(it / thin) = (x - y).square().sum();
        }

        // Stop if coalesced
        if((x == y).all())
        {
            tau = it;
            break;
        }
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("tau") = tau);
}
