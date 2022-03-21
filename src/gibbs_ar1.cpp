/*
Gibbs samplers for the periodic AR(1) process of D. Wilkison, in the discussion of Jacob et al. - Unbiased MCMC with couplings (JRSSB, 2021)
*/
// The samplers only work for d >= 2. 

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "rng.h"

using Eigen::ArrayXd;

using std::vector;

/*
    // Precomputed terms:
    // for conditional means
    ArrayXd q_inv = Q.diagonal().array().inverse(); // Vector of inverses of diagonal entries of Q
    ArrayXd a = q_inv * (Q * mu).array();
    MatrixXd A = MatrixXd::Identity(d, d) - q_inv.matrix().asDiagonal() * Q;

    // for conditional standard deviations
    ArrayXd sd = q_inv.sqrt();
*/

// TO DO: rewrite for output to R
//
//

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List SimulateGibbs_ar1(const Eigen::Map<Eigen::ArrayXd> &x0, 
                            const double &sd,
                            const double &c, 
                            const int &iter_store_x,
                            const int &iter_final_x,
                            const int &thin)
{
    int d = x0.rows();

    // Storage
    vector<ArrayXd> xs(iter_store_x / thin + 1);
    ArrayXd x_final(d);

    // Initial state
    ArrayXd x = x0;

    xs[0] = x;

    // Gibbs steps
    for (int i = 1; i <= iter_final_x; i++)
    {
        x(0) = c * (x(d - 1) + x(1)) + RNG::rnorm(RNG::rng) * sd;
        for (int j = 1; j < d - 1; j++)
        {
            x(j) = c * (x(j - 1) + x(j + 1)) + RNG::rnorm(RNG::rng) * sd; // + a(j)
        }
        x(d - 1) = c * (x(d - 2) + x(0)) + RNG::rnorm(RNG::rng) * sd;

        // Storage, thinned
        if (i % thin == 0 && i <= iter_store_x)
        {
            xs[i/thin] = x;
        }
        if (i == iter_final_x)
        {
            x_final = x;
        }
    }

    return Rcpp::List::create(Rcpp::Named("x") = Rcpp::wrap(xs),
                              Rcpp::Named("x_final") = Rcpp::wrap(x_final));
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List SimulateReflMaxGibbs_ar1(const Eigen::Map<Eigen::ArrayXd> &x0, 
                                    const Eigen::Map<Eigen::ArrayXd> &y0,
                                    const double &sd,
                                    const double &c,
                                    const int &L,
                                    const int &iter_store_x,
                                    const int &iter_final_x,
                                    const int &thin)
{
    // Constants
    int d = x0.rows();
    double var = sd * sd;

    // Temporary variables
    double z, mu_x, mu_y, log_cpl;

    // Storage
    vector<ArrayXd> xs(iter_store_x / thin + 1);
    vector<double> xy_lagged_square_dist;
    ArrayXd x_ref;
    int tau = -1;

    // Initial states
    ArrayXd x = x0;
    ArrayXd y = y0;

    xs[0] = x;

    // L steps for X
    for (int i = 1; i <= L; i++)
    {
        x(0) = c * (x(d - 1) + x(1)) + RNG::rnorm(RNG::rng) * sd;
        for (int j = 1; j < d - 1; j++)
        {
            x(j) = c * (x(j - 1) + x(j + 1)) + RNG::rnorm(RNG::rng) * sd;
        }
        x(d - 1) = c * (x(d - 2) + x(0)) + RNG::rnorm(RNG::rng) * sd;

        // Storage
        if ((i % thin == 0) && (i <= iter_store_x))
        {
            xs[i / thin] = x;
        }
        if (i == iter_final_x)
        {
            x_ref = x;
        }
    }

    // Coupled steps
    int i = L;
    while(tau == -1) 
    {
        xy_lagged_square_dist.push_back((x - y).square().sum()); // This needs to be the squared norm. Will be of length tau - L, its last entry should be non-zero.
        
        i+=1;
    
        z = sd * RNG::rnorm(RNG::rng);
        mu_x = c * (x(d - 1) + x(1));
        mu_y = c * (y(d - 1) + y(1));

        x(0) = mu_x + z;

        log_cpl = 0.5 * (z * z - (x(0) - mu_y) * (x(0) - mu_y)) / var;
        if (log_cpl > 0 || log(RNG::runif(RNG::rng)) < log_cpl)
        {
            y(0) = x(0);
        } else
        {
            y(0) = mu_y - z;
        }
        for (int j = 1; j < d - 1; j++)
        {
            z = sd * RNG::rnorm(RNG::rng);
            mu_x = c * (x(j - 1) + x(j + 1));
            mu_y = c * (y(j - 1) + y(j + 1));

            x(j) = mu_x + z;

            log_cpl = 0.5 * (z * z - (x(j) - mu_y) * (x(j) - mu_y)) / var;
            if (log_cpl > 0 || log(RNG::runif(RNG::rng)) < log_cpl)
            {
                y(j) = x(j);
            } else
            {
                y(j) = mu_y - z;
            }
        }

        z = sd * RNG::rnorm(RNG::rng);
        mu_x = c * (x(d - 2) + x(0));
        mu_y = c * (y(d - 2) + y(0));

        x(d - 1) = mu_x + z;

        log_cpl = 0.5 * (z * z - (x(d - 1) - mu_y) * (x(d - 1) - mu_y)) / var;
        if (log_cpl > 0 || log(RNG::runif(RNG::rng)) < log_cpl)
        {
            y(d - 1) = x(d - 1);
        } else
        {
            y(d - 1) = mu_y - z;
        }

        // Storage
        if ((i % thin == 0) && (i <= iter_store_x))
        {
            xs[i / thin] = x;
        }
        if (i == iter_final_x)
        {
            x_ref = x;
        }

        if(x.isApprox(y)) {
            tau = i;
        }
    }

    // Additional steps, if needed
    if (tau <= iter_final_x)
    {
        for (int i = tau + 1; i <= iter_final_x; i++)
        {
            x(0) = c * (x(d - 1) + x(1)) + RNG::rnorm(RNG::rng) * sd;
            for (int j = 1; j < d - 1; j++)
            {
                x(j) = c * (x(j - 1) + x(j + 1)) + RNG::rnorm(RNG::rng) * sd;
            }
            x(d - 1) = c * (x(d - 2) + x(0)) + RNG::rnorm(RNG::rng) * sd;

            // Storage
            if ((i % thin == 0) && (i <= iter_store_x))
            {
                xs[i / thin] = x;
            }
            if (i == iter_final_x)
            {
                x_ref = x;
            }
        }
    }

    return Rcpp::List::create(Rcpp::Named("x") = xs,
                              Rcpp::Named("x_final") = x_ref,
                              Rcpp::Named("squaredist") = xy_lagged_square_dist,
                              Rcpp::Named("tau") = tau);
}
