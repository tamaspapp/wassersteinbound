// Samplers for the stochastic volatility model

#include <RcppEigen.h>
#include <omp.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11, openmp)]]

#include "rng.h"

// Function to sample the latent variables from the model
//
//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::NumericVector SampleLatentVariables(int T, double sig, double phi) {
  
    Rcpp::NumericVector x(T);
  
    x(0) = sqrt(sig * sig / (1. - phi * phi)) * RNG::rnorm(RNG::rng);

    for(int i = 1; i < T; i++)
    {
        x(i) = phi * x(i - 1) + sig * RNG::rnorm(RNG::rng);
    }
  
  return x;
}

// Target log-density
inline double logpi_nograd(const Eigen::Ref<const Eigen::ArrayXd> &x,
                           const Eigen::Ref<const Eigen::ArrayXd> &y_sq,
                           const double &inv_beta_sq,
                           const double &phi, const double &phi_sq, const double &inv_phi_sq,
                           const double &inv_sigma_sq,
                           const int &T)
{
    // "temp" is (a constant shift off) from twice the negative log-posterior
    double temp = x.sum() + inv_beta_sq  * (y_sq * exp(-x)).sum();

    temp += inv_sigma_sq * (phi * x.segment(0, T - 2) -  x.segment(1, T - 1)).square().sum();

    temp += (1 - phi_sq) * inv_sigma_sq * x(0) * x(0);

    return (-0.5) * temp;
}

// Target log-density
inline double logpi(const Eigen::Ref<const Eigen::ArrayXd> &x,
                    const Eigen::Ref<const Eigen::ArrayXd> &exp_minus_x,
                    const Eigen::Ref<const Eigen::ArrayXd> &x_diff, // Entries are: phi * x_{t} - x_{t+1}
                    const Eigen::Ref<const Eigen::ArrayXd> &y_sq,
                    const double &inv_beta_sq,
                    const double &phi, const double &phi_sq, const double &inv_phi_sq,
                    const double &inv_sigma_sq,
                    const int &T)
{
    // "temp" is (a constant off) from twice the negative log-posterior
    double temp = x.sum() + inv_beta_sq  * (y_sq * exp_minus_x).sum();

    temp += inv_sigma_sq * x_diff.square().sum();

    temp += (1 - phi_sq) * inv_sigma_sq * x(0) * x(0);

    return (-0.5) * temp;
}

// Target log-gradient
inline Eigen::ArrayXd gradlogpi(const Eigen::Ref<const Eigen::ArrayXd> &x,
                         const Eigen::Ref<const Eigen::ArrayXd> &exp_minus_x,
                         const Eigen::Ref<const Eigen::ArrayXd> &x_diff, // Entries are: phi * x_{t} - x_{t+1}
                         const Eigen::Ref<const Eigen::ArrayXd> &y_sq,
                         const double &inv_beta_sq,
                         const double &phi, const double &phi_sq, const double &inv_phi_sq,
                         const double &inv_sigma_sq,
                         const int &T)
{
    // "temp" is the log-posterior-gradient
    Eigen::ArrayXd temp = 0.5 * inv_beta_sq * (y_sq * exp_minus_x);

    temp(0) -= (1 - phi_sq) * inv_sigma_sq * x(0);

    temp.segment(0, T - 2) -= phi * inv_sigma_sq * x_diff;

    temp.segment(1, T - 1) += inv_sigma_sq * x_diff;

    return temp;
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateRWM_SVM(const Eigen::Map<Eigen::ArrayXd> x0,
                 const Eigen::Map<Eigen::ArrayXd> y,
                 const double &beta,
                 const double &sigma,
                 const double &phi,
                 const double &h,
                 const int &iter_store_x,
                 const int &iter_final_x,
                 const int &thin)
{
    // Declare constants
    double beta_sq     = beta * beta;
    double inv_beta_sq = 1 / beta_sq;

    double phi_sq     = phi * phi;
    double inv_phi_sq = 1 / phi_sq;

    double sigma_sq     = sigma * sigma;
    double inv_sigma_sq = 1 / sigma_sq;

    int T = x0.rows();
    Eigen::ArrayXd y_sq = y.square();

    // Declare temporary variables
    double logpi_x, logpi_xp;
    double logHR_x;
    double u;

    Eigen::ArrayXd x(T), xp(T);
    Eigen::ArrayXd x_dot(T);

    // Intialize storage
    std::vector<Eigen::ArrayXd> xs(iter_store_x / thin + 1); // C++ integer division use
    Eigen::ArrayXd x_final(T);

    x = x0;
    logpi_x = logpi_nograd(x, y_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    xs[0] = x;

    int accepted = 0;

    // "iter" iterations of MCMC
    for (int i = 1; i <= iter_final_x; i++)
    {
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x + h * x_dot;
        logpi_xp = logpi_nograd(xp, y_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Compute log of Hastings ratio
        logHR_x = logpi_xp - logpi_x;

        // Perform acceptance step
        u = RNG::runif(RNG::rng);
        if (logHR_x > 0 || log(u) < logHR_x)
        {
            ++accepted;
            x = xp;
            logpi_x = logpi_xp;
        }

        // Store
        if (i % thin == 0 && i <= iter_store_x)
        {
            xs[i / thin] = x;
        }
        if (i == iter_final_x)
        {
            x_final = x;
        }
    }
    return Rcpp::List::create(Rcpp::Named("x") = Rcpp::wrap(xs),
                              Rcpp::Named("x_final") = Rcpp::wrap(x_final),
                              Rcpp::Named("acc_rate") = (double) accepted / iter_final_x);
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateReflMaxRWM_SVM(const Eigen::Map<Eigen::ArrayXd> x0,
                              const Eigen::Map<Eigen::ArrayXd> y0,
                              const Eigen::Map<Eigen::ArrayXd> y_data,
                              const double &beta,
                              const double &sigma,
                              const double &phi,
                              const double &h,
                              const int &L,
                              const int &iter_store_x,
                              const int &iter_final_x,
                              const int &thin)
{
    // Declare constants
    double beta_sq     = beta * beta;
    double inv_beta_sq = 1 / beta_sq;

    double phi_sq     = phi * phi;
    double inv_phi_sq = 1 / phi_sq;

    double sigma_sq     = sigma * sigma;
    double inv_sigma_sq = 1 / sigma_sq;

    int T = x0.rows();
    Eigen::ArrayXd y_data_sq = y_data.square();

    // Declare temporary variables
    double log_u, logcpl, ucpl, z_sqnorm;
    Eigen::ArrayXd z(T);

    double logpi_x, logpi_xp;
    double logHR_x;
    bool acc_x;

    Eigen::ArrayXd x(T), xp(T);
    Eigen::ArrayXd x_dot(T);

    double logpi_y, logpi_yp;
    double logHR_y;
    bool acc_y;

    Eigen::ArrayXd y(T), yp(T);
    Eigen::ArrayXd y_dot(T);
    
    // Intialize storage
    int tau = -1;
    std::vector<Eigen::ArrayXd> xs(iter_store_x / thin + 1); // C++ integer division use
    Eigen::ArrayXd x_final(T);
    std::vector<double> xy_lagged_square_dist;

    x = x0;
    logpi_x = logpi_nograd(x, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    xs[0] = x;

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x + h * x_dot;
        logpi_xp = logpi_nograd(xp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Compute log of Hastings ratio
        logHR_x = logpi_xp - logpi_x;

        // Perform acceptance step
        log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
        }

        if (i % thin == 0 && i <= iter_store_x)
        {
            xs[i / thin] = x;
        }

        if (i == iter_final_x)
        {
            x_final = x;
        } 
    }

    y = y0;
    logpi_y = logpi_nograd(y, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    int i = L;

    bool coupled = false;
    // Iterations until coupling
    while (tau == -1)
    {
        if ((i - L) % thin == 0)
        {
            xy_lagged_square_dist.push_back((x - y).square().sum());
        }

        i += 1;

        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x + h * x_dot;
        logpi_xp = logpi_nograd(xp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Compute log of Hastings ratio
        logHR_x = logpi_xp - logpi_x;

        // Reflection-maximal coupling for Y
        z = (x - y) / h;

        logcpl = -0.5 * (x_dot + z).square().sum() + 0.5 * x_dot.square().sum();
        ucpl = RNG::runif(RNG::rng);
        coupled = false;

        log_u = log(RNG::runif(RNG::rng));

        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling of proposals
        {
            coupled = true;
            yp = xp;
            logpi_yp = logpi_xp;
        }
        else // Reflect
        {
            z_sqnorm = z.square().sum();
            yp = y + h * (x_dot - 2 / z_sqnorm * (z * x_dot).sum() * z);

            logpi_yp = logpi_nograd(yp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        }

        logHR_y = logpi_yp - logpi_y;

        
        // Accept-reject
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            acc_x = true;
        } else { acc_x = false; }

        
        if (logHR_y > 0 || log_u < logHR_y)
        {
            y = yp;
            logpi_y = logpi_yp;
            acc_y = true;
        } else { acc_y = false; }

        // Store
        if (i % thin == 0 && i <= iter_store_x)
        {
            xs[i / thin] = x;
        }
        if (i == iter_final_x)
        {
            x_final = x;
        }

        // Stop if coupled
        if (acc_x * acc_y * coupled == true)
        {
            tau = i;
            break;
        } 
    }

    // Final iterations after coupling
    if (tau <= iter_final_x)
    {
        for (int i = tau + 1; i <= iter_final_x; i++)
        {
            // Proposal noise
            for (int j = 0; j < T; j++)
            {
                x_dot(j) = RNG::rnorm(RNG::rng);
            }
            xp = x + h * x_dot;
            logpi_xp = logpi_nograd(xp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

            // Compute log of Hastings ratio
            logHR_x = logpi_xp - logpi_x;

            // Perform acceptance step
            log_u = log(RNG::runif(RNG::rng));
            if (logHR_x > 0 || log_u < logHR_x)
            {
                x = xp;
                logpi_x = logpi_xp;
            }

            // Store
            if (i % thin == 0 && i <= iter_store_x)
            {
                xs[i / thin] = x;
            }
        
            if (i == iter_final_x)
            {
                x_final = x;
            }
        }
    }

    return Rcpp::List::create(Rcpp::Named("x") = Rcpp::wrap(xs),
                              Rcpp::Named("x_final") = Rcpp::wrap(x_final),
                              Rcpp::Named("squaredist") = Rcpp::wrap(xy_lagged_square_dist),
                              Rcpp::Named("tau") = tau);
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateMALA_SVM(const Eigen::Map<Eigen::ArrayXd> x0,
                  const Eigen::Map<Eigen::ArrayXd> y,
                  const double &beta,
                  const double &sigma,
                  const double &phi,
                  const double &h,
                  const int &iter_store_x,
                  const int &iter_final_x,
                  const int &thin)
{
    // Declare constants
    double h_sq     = h * h;
    double inv_h_sq = 1 / h_sq;
    
    double beta_sq     = beta * beta;
    double inv_beta_sq = 1 / beta_sq;

    double phi_sq     = phi * phi;
    double inv_phi_sq = 1 / phi_sq;

    double sigma_sq     = sigma * sigma;
    double inv_sigma_sq = 1 / sigma_sq;

    int T = x0.rows();
    Eigen::ArrayXd y_sq = y.square();

    // Declare temporary variables
    double logpi_x, logpi_xp;
    double logq_x_xp, logq_xp_x;
    double logHR_x;
    double u;

    Eigen::ArrayXd x(T), xp(T);
    Eigen::ArrayXd x_dot(T);
    Eigen::ArrayXd x_mean(T), xp_mean(T);
    Eigen::ArrayXd exp_minus_xp(T);
    Eigen::ArrayXd xp_diff(T - 1);
    
    // Intialize storage
    std::vector<Eigen::ArrayXd> xs(iter_store_x / thin + 1); // C++ integer division use
    Eigen::ArrayXd x_final(T);

    x = x0;
    exp_minus_xp = (-x).exp();
    xp_diff = phi * x.segment(0, T - 2) -  x.segment(1, T - 1);
    logpi_x = logpi(x, exp_minus_xp, xp_diff, y_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
    x_mean = x + 0.5 * h_sq * gradlogpi(x, exp_minus_xp, xp_diff, y_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    xs[0] = x;

    int accepted = 0;

    // "iter" iterations of MCMC
    for (int i = 1; i <= iter_final_x; i++)
    {
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        exp_minus_xp = (-xp).exp();
        xp_diff = phi * xp.segment(0, T - 2) -  xp.segment(1, T - 1);

        logpi_xp = logpi(xp, exp_minus_xp, xp_diff, y_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, exp_minus_xp, xp_diff, y_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).square().sum(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.square().sum();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Perform acceptance step
        u = RNG::runif(RNG::rng);
        if (logHR_x > 0 || log(u) < logHR_x)
        {
            ++accepted;
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        }

        // Store
        if (i % thin == 0 && i <= iter_store_x)
        {
            xs[i / thin] = x;
        }
        if (i == iter_final_x)
        {
            x_final = x;
        }
    }
    return Rcpp::List::create(Rcpp::Named("x") = Rcpp::wrap(xs),
                              Rcpp::Named("x_final") = Rcpp::wrap(x_final),
                              Rcpp::Named("acc_rate") = (double) accepted / iter_final_x);
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateReflMaxMALA_SVM(const Eigen::Map<Eigen::ArrayXd> x0,
                         const Eigen::Map<Eigen::ArrayXd> y0,
                         const Eigen::Map<Eigen::ArrayXd> y_data,
                         const double &beta,
                         const double &sigma,
                         const double &phi,
                         const double &h,
                         const int &L,
                         const int &iter_store_x,
                         const int &iter_final_x,
                         const int &thin)
{

    // Declare constants
    double h_sq     = h * h;
    double inv_h_sq = 1 / h_sq;
    
    double beta_sq     = beta * beta;
    double inv_beta_sq = 1 / beta_sq;

    double phi_sq     = phi * phi;
    double inv_phi_sq = 1 / phi_sq;

    double sigma_sq     = sigma * sigma;
    double inv_sigma_sq = 1 / sigma_sq;

    int T = x0.rows();
    Eigen::ArrayXd y_data_sq = y_data.square();

    // Declare temporary variables
    double log_u;
    Eigen::ArrayXd z(T);
    double z_sqnorm;
    double ucpl, logcpl;

    double logpi_x, logpi_xp;
    double logq_x_xp, logq_xp_x;
    double logHR_x;
    bool acc_x;

    Eigen::ArrayXd x(T), xp(T);
    Eigen::ArrayXd x_dot(T);
    Eigen::ArrayXd x_mean(T), xp_mean(T);
    Eigen::ArrayXd exp_minus_xp(T);
    Eigen::ArrayXd xp_diff(T - 1);

    double logpi_y, logpi_yp;
    double logq_y_yp, logq_yp_y;
    double logHR_y;
    bool acc_y;

    Eigen::ArrayXd y(T), yp(T);
    Eigen::ArrayXd y_dot(T);
    Eigen::ArrayXd y_mean(T), yp_mean(T);
    Eigen::ArrayXd exp_minus_yp(T);
    Eigen::ArrayXd yp_diff(T - 1);
    
    // Intialize storage
    int tau = -1;
    std::vector<Eigen::ArrayXd> xs(iter_store_x / thin + 1); // C++ integer division use
    Eigen::ArrayXd x_final(T);
    std::vector<double> xy_lagged_square_dist;

    // Start MCMC
    x = x0;
    exp_minus_xp = (-x).exp();
    xp_diff = phi * x.segment(0, T - 2) -  x.segment(1, T - 1);
    logpi_x = logpi(x, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
    x_mean = x + 0.5 * h_sq * gradlogpi(x, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    xs[0] = x;

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        exp_minus_xp = (-xp).exp();
        xp_diff = phi * xp.segment(0, T - 2) -  xp.segment(1, T - 1);

        logpi_xp = logpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).square().sum(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.square().sum();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Perform acceptance step
        log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        }

            // Store
            if (i % thin == 0 && i <= iter_store_x)
            {
                xs[i / thin] = x;
            }
            if (i == iter_final_x)
            {
                x_final = x;
            }
    }

    y = y0;
    exp_minus_yp = (-y).exp();
    yp_diff = phi * y.segment(0, T - 2) -  y.segment(1, T - 1);
    logpi_y = logpi(y, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
    y_mean = y + 0.5 * h_sq * gradlogpi(y, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
    
    int i = L;

    bool coupled = false;
    // Iterations until coupling
    while (tau == -1)
    {
        if ((i - L) % thin == 0)
        {
            xy_lagged_square_dist.push_back((x - y).square().sum());
        }

        i += 1;

        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        exp_minus_xp = (-xp).exp();
        xp_diff = phi * xp.segment(0, T - 2) -  xp.segment(1, T - 1);

        logpi_xp = logpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).square().sum(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.square().sum();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Reflection-maximal coupling for Y
        z = (x_mean - y_mean) / h;

        logcpl = -0.5 * (x_dot + z).square().sum() + 0.5 * x_dot.square().sum();
        ucpl = RNG::runif(RNG::rng);
        coupled = false;

        log_u = log(RNG::runif(RNG::rng));

        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling of proposals
        {
            coupled = true;
            yp = xp;
            yp_mean = xp_mean;

            logpi_yp = logpi_xp;
            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).square().sum(); // Go from yp = xp to y
            logq_y_yp = -0.5 * inv_h_sq * (yp - y_mean).square().sum(); // Go from y to yp = xp
        }
        else // Reflect
        {
            z_sqnorm = z.square().sum();
            yp = y_mean + h * (x_dot - 2 / z_sqnorm * (z * x_dot).sum() * z);

            exp_minus_yp = (-yp).exp();
            yp_diff = phi * yp.segment(0, T - 2) -  yp.segment(1, T - 1);

            yp_mean = yp + 0.5 * h_sq * gradlogpi(yp, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

            logpi_yp = logpi(yp, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);            
            // Proposal log-densities
            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).square().sum(); // Go from yp to y
            logq_y_yp = logq_x_xp;                                      // Go from y to yp

        }

        logHR_y = logpi_yp + logq_yp_y - logpi_y - logq_y_yp;

        // Accept-reject
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
            acc_x = true;
        } else { acc_x = false; }

        if (logHR_y > 0 || log_u < logHR_y)
        {
            y = yp;
            logpi_y = logpi_yp;
            y_mean = yp_mean;
            acc_y = true;
        } else { acc_y = false; }

        // Store
        if (i % thin == 0 && i <= iter_store_x)
        {
            xs[i / thin] = x;
        }
        if (i == iter_final_x)
        {
            x_final = x;
        }

        // Stop if coupled
        if (acc_x * acc_y * coupled == true)
        {
            tau = i;
            break;
        } 
    }

    // Final iterations after coupling
    if (tau <= iter_final_x)
    {
        for (int i = tau + 1; i <= iter_final_x; i++)
        {
            // Proposal noise
            for (int j = 0; j < T; j++)
            {
                x_dot(j) = RNG::rnorm(RNG::rng);
            }
            xp = x_mean + h * x_dot;
            exp_minus_xp = (-xp).exp();
            xp_diff = phi * xp.segment(0, T - 2) -  xp.segment(1, T - 1);

            logpi_xp = logpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
            xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

            // Proposal log-densities
            logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).square().sum(); // Go from xp to x
            logq_x_xp = -0.5 * x_dot.square().sum();                    // Go from x to xp

            // Compute log of Hastings ratio
            logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

            // Perform acceptance step
            log_u = log(RNG::runif(RNG::rng));
            if (logHR_x > 0 || log_u < logHR_x)
            {
                x = xp;
                logpi_x = logpi_xp;
                x_mean = xp_mean;
            }

            // Store
            if (i % thin == 0 && i <= iter_store_x)
            {
                xs[i / thin] = x;
            }
            if (i == iter_final_x)
            {
                x_final = x;
            }
        }
    }
    
    return Rcpp::List::create(Rcpp::Named("x") = Rcpp::wrap(xs),
                              Rcpp::Named("x_final") = Rcpp::wrap(x_final),
                              Rcpp::Named("squaredist") = Rcpp::NumericVector(xy_lagged_square_dist.begin(), xy_lagged_square_dist.end()),
                              Rcpp::Named("tau") = tau);
}


//' @export
//[[Rcpp::export(rng = false)]]
int SimulateReflMaxRWMMeetingTime_SVM(const Eigen::Map<Eigen::ArrayXd> x0,
                                   const Eigen::Map<Eigen::ArrayXd> y0,
                                   const Eigen::Map<Eigen::ArrayXd> y_data,
                                   const double &beta,
                                   const double &sigma,
                                   const double &phi,
                                   const double &h,
                                   const int &L,
                                   const int &maxiter)
{
    // Declare constants
    double beta_sq     = beta * beta;
    double inv_beta_sq = 1 / beta_sq;

    double phi_sq     = phi * phi;
    double inv_phi_sq = 1 / phi_sq;

    double sigma_sq     = sigma * sigma;
    double inv_sigma_sq = 1 / sigma_sq;

    int T = x0.rows();
    Eigen::ArrayXd y_data_sq = y_data.square();

    // Declare temporary variables
    double log_u, logcpl, ucpl, z_sqnorm;
    Eigen::ArrayXd z(T);

    double logpi_x, logpi_xp;
    double logHR_x;
    bool acc_x;

    Eigen::ArrayXd x(T), xp(T);
    Eigen::ArrayXd x_dot(T);

    double logpi_y, logpi_yp;
    double logHR_y;
    bool acc_y;

    Eigen::ArrayXd y(T), yp(T);
    Eigen::ArrayXd y_dot(T);
    
    // Intialize storage
    int tau = -1;

    x = x0;
    logpi_x = logpi_nograd(x, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x + h * x_dot;
        logpi_xp = logpi_nograd(xp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Compute log of Hastings ratio
        logHR_x = logpi_xp - logpi_x;

        // Perform acceptance step
        log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
        }
    }

    y = y0;
    logpi_y = logpi_nograd(y, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    // Iterate until coupling
    int i = L;
    bool coupled = false;

    while(tau == -1 && i <= maxiter)
    {
        i +=1;
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x + h * x_dot;
        logpi_xp = logpi_nograd(xp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Compute log of Hastings ratio
        logHR_x = logpi_xp - logpi_x;

        // Reflection-maximal coupling for Y
        z = (x - y) / h;

        logcpl = -0.5 * (x_dot + z).square().sum() + 0.5 * x_dot.square().sum();
        ucpl = RNG::runif(RNG::rng);
        coupled = false;

        log_u = log(RNG::runif(RNG::rng));

        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling of proposals
        {
            coupled = true;
            yp = xp;
            logpi_yp = logpi_xp;

        }
        else // Reflect
        {
            z_sqnorm = z.square().sum();
            yp = y + h * (x_dot - 2 / z_sqnorm * (z * x_dot).sum() * z);

            logpi_yp = logpi_nograd(yp, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        }

        logHR_y = logpi_yp - logpi_y;

        // Accept-reject
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            acc_x = true;
        } else { acc_x = false; }

        if (logHR_y > 0 || log_u < logHR_y)
        {
            y = yp;
            logpi_y = logpi_yp;
            acc_y = true;
        } else { acc_y = false; }

        // Stop if coupled
        if (acc_x * acc_y * coupled == true)
        {
            tau = i;
            break;
        } 
    }

    return tau;
}


//' @export
//[[Rcpp::export(rng = false)]]
int SimulateReflMaxMALAMeetingTime_SVM(const Eigen::Map<Eigen::ArrayXd> x0,
                                   const Eigen::Map<Eigen::ArrayXd> y0,
                                   const Eigen::Map<Eigen::ArrayXd> y_data,
                                   const double &beta,
                                   const double &sigma,
                                   const double &phi,
                                   const double &h,
                                   const int &L)
{

    // Declare constants
    double h_sq     = h * h;
    double inv_h_sq = 1 / h_sq;
    
    double beta_sq     = beta * beta;
    double inv_beta_sq = 1 / beta_sq;

    double phi_sq     = phi * phi;
    double inv_phi_sq = 1 / phi_sq;

    double sigma_sq     = sigma * sigma;
    double inv_sigma_sq = 1 / sigma_sq;

    int T = x0.rows();
    Eigen::ArrayXd y_data_sq = y_data.square();

    // Declare temporary variables
    double log_u;
    Eigen::ArrayXd z(T);
    double z_sqnorm;
    double ucpl, logcpl;

    double logpi_x, logpi_xp;
    double logq_x_xp, logq_xp_x;
    double logHR_x;
    bool acc_x;

    Eigen::ArrayXd x(T), xp(T);
    Eigen::ArrayXd x_dot(T);
    Eigen::ArrayXd x_mean(T), xp_mean(T);
    Eigen::ArrayXd exp_minus_xp(T);
    Eigen::ArrayXd xp_diff(T - 1);

    double logpi_y, logpi_yp;
    double logq_y_yp, logq_yp_y;
    double logHR_y;
    bool acc_y;

    Eigen::ArrayXd y(T), yp(T);
    Eigen::ArrayXd y_dot(T);
    Eigen::ArrayXd y_mean(T), yp_mean(T);
    Eigen::ArrayXd exp_minus_yp(T);
    Eigen::ArrayXd yp_diff(T - 1);
    
    // Intialize storage
    int tau = -1;

    // Start MCMC
    x = x0;
    exp_minus_xp = (-x).exp();
    xp_diff = phi * x.segment(0, T - 2) -  x.segment(1, T - 1);
    logpi_x = logpi(x, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
    x_mean = x + 0.5 * h_sq * gradlogpi(x, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        exp_minus_xp = (-xp).exp();
        xp_diff = phi * xp.segment(0, T - 2) -  xp.segment(1, T - 1);

        logpi_xp = logpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).square().sum(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.square().sum();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Perform acceptance step
        log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        }
    }

    y = y0;
    exp_minus_yp = (-y).exp();
    yp_diff = phi * y.segment(0, T - 2) -  y.segment(1, T - 1);
    logpi_y = logpi(y, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
    y_mean = y + 0.5 * h_sq * gradlogpi(y, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

    
    // Iterations until couple
    int i = L;
    bool coupled = false;

    while (tau == -1)
    {
        i +=1;
        // Proposal noise
        for (int j = 0; j < T; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        exp_minus_xp = (-xp).exp();
        xp_diff = phi * xp.segment(0, T - 2) -  xp.segment(1, T - 1);

        logpi_xp = logpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(xp, exp_minus_xp, xp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).square().sum(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.square().sum();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Reflection-maximal coupling for Y
        z = (x_mean - y_mean) / h;

        logcpl = -0.5 * (x_dot + z).square().sum() + 0.5 * x_dot.square().sum();
        ucpl = RNG::runif(RNG::rng);
        coupled = false;

        log_u = log(RNG::runif(RNG::rng));

        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling of proposals
        {
            coupled = true;
            yp = xp;
            yp_mean = xp_mean;

            logpi_yp = logpi_xp;
            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).square().sum();
            logq_y_yp = -0.5 * inv_h_sq * (yp - y_mean).square().sum();

        }
        else // Reflect
        {
            z_sqnorm = z.square().sum();
            yp = y_mean + h * (x_dot - 2 / z_sqnorm * (z * x_dot).sum() * z);

            exp_minus_yp = (-yp).exp();
            yp_diff = phi * yp.segment(0, T - 2) -  yp.segment(1, T - 1);

            yp_mean = yp + 0.5 * h_sq * gradlogpi(yp, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);

            logpi_yp = logpi(yp, exp_minus_yp, yp_diff, y_data_sq, inv_beta_sq, phi, phi_sq, inv_phi_sq, inv_sigma_sq, T);
            // Proposal log-densities
            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).square().sum(); // Go from yp to y
            logq_y_yp = logq_x_xp;                                      // Go from y to yp

        }

        logHR_y = logpi_yp + logq_yp_y - logpi_y - logq_y_yp;

        // Accept-reject
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
            acc_x = true;
        } else { acc_x = false; }

        if (logHR_y > 0 || log_u < logHR_y)
        {
            y = yp;
            logpi_y = logpi_yp;
            y_mean = yp_mean;
            acc_y = true;
        } else { acc_y = false; }

        // Stop if coupled
        if (acc_x * acc_y * coupled == true)
        {
            tau = i;
            break;
        } 
    }
    
    return tau;
}