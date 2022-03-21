// Samplers for the MALA vs ULA scaling experiment

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "rng.h"

// Target log-density: pi = N(0, Sigma), where Sigma = t(U) %*% U
inline double logpi(const Eigen::Ref<const Eigen::SparseMatrix<double>> &U, const Eigen::Ref<const Eigen::VectorXd> &x)
{
    return -0.5 * (U * x).squaredNorm();
}

// Target log-gradient: f(x) = - Sigma^(-1) %*% x
inline Eigen::VectorXd gradlogpi(const Eigen::Ref<const Eigen::SparseMatrix<double>> &Sigma_inv, const Eigen::Ref<const Eigen::VectorXd> &x)
{
    return -Sigma_inv * x;
}


// "iter" = maximum number of iterations for the X-chain

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateULA_scaling(const Eigen::Map<Eigen::SparseMatrix<double>> &Sigma_inv,
                               const Eigen::Map<Eigen::SparseMatrix<double>> &U,
                               const double &disp,
                               const double &h,
                               const int &iter, const int &thin)
{
    // Declare constants
    int d = Sigma_inv.rows();
    double h_sq = h * h;

    // Declare temporary variables
    Eigen::VectorXd x(d);
    Eigen::VectorXd x_dot(d);

    // Intialize storage
    std::vector<Eigen::VectorXd> xs(iter / thin + 1); // C++ integer division used

    // Initial state pi_0 = N(mean, disp^2 I)
    for (int i = 0; i < d; i++)
    {
        x(i) = RNG::rnorm(RNG::rng);
    }
    x = disp * x;

    xs[0] = x;

    // "iter" iterations of MCMC
    for (int i = 1; i <= iter; i++)
    {
        // Generate proposal
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        x += 0.5 * h_sq * gradlogpi(Sigma_inv, x) + h * x_dot;

        if (i % thin == 0)
        {
            xs[i / thin] = x;
        }
    }

    return Rcpp::wrap(xs);
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateMALA_scaling(const Eigen::Map<Eigen::SparseMatrix<double>> &Sigma_inv,
                  const Eigen::Map<Eigen::SparseMatrix<double>> &U,
                  const double &disp,
                  const double &h,
                  const int &iter, const int &thin)
{
    // Declare constants
    double h_sq = h * h;
    double inv_h_sq = 1 / h_sq;
    int d = Sigma_inv.rows();

    // Declare temporary variables
    double logpi_x, logpi_xp;
    double logq_x_xp, logq_xp_x;
    double logHR_x;
    double u;

    Eigen::VectorXd x(d), xp(d);
    Eigen::VectorXd x_mean(d), xp_mean(d);
    Eigen::VectorXd x_dot(d);

    // Intialize storage
    std::vector<Eigen::VectorXd> xs(iter / thin + 1); // C++ integer division used
    std::vector<bool> acc_x(iter);

    // Initial state pi_0 = N(mean, disp^2 I)
    for (int i = 0; i < d; i++)
    {
        x(i) = RNG::rnorm(RNG::rng);
    }
    x = disp * x;
    logpi_x = logpi(U, x);
    x_mean = x + 0.5 * h_sq * gradlogpi(Sigma_inv, x);

    xs[0] = x;

    // "iter" iterations of MCMC
    for (int i = 1; i <= iter; i++)
    {
        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;

        logpi_xp = logpi(U, xp);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Perform acceptance step
        u = RNG::runif(RNG::rng);
        if (logHR_x > 0 || log(u) < logHR_x)
        {
            acc_x[i - 1] = true;
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        }

        if (i % thin == 0)
        {
            xs[i / thin] = x;
        }
    }
    return Rcpp::List::create(Rcpp::Named("x") = Rcpp::wrap(xs), Rcpp::Named("acc_x") = Rcpp::wrap(acc_x));
}

// "iter" = maximum number of iterations for the X-chain, always

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateReflMaxULA_scaling(const Eigen::Map<Eigen::SparseMatrix<double>> &Sigma_inv,
                            const Eigen::Map<Eigen::SparseMatrix<double>> &U,
                            const double &disp,
                            const double &h,
                            const int &L,
                            const int &iter,           // Iterations for the main part of the bound
                            const int &iter_reference, // Iteration to use as a reference for the debiasing
                            const int &thin)
{
    // Declare temporary variables
    double h_sq = h * h;
    int d = Sigma_inv.rows();
    

    Eigen::VectorXd x(d), y(d);
    Eigen::VectorXd x_mean(d), y_mean(d);
    Eigen::VectorXd x_dot(d);
    Eigen::VectorXd z(d);
    
    double ucpl, logcpl;

    double z_sqnorm;

    // Intialize storage
    std::vector<Eigen::VectorXd> xs(iter / thin + 1); // C++ integer division used
    Eigen::VectorXd x_reference(d);
    int tau = -1;
    std::vector<double> w2_bound_parts;

    // Initial state pi_0 = N(mean, disp^2 I)
    for (int i = 0; i < d; i++)
    {
        x(i) = RNG::rnorm(RNG::rng);
    }
    x = disp * x;
    xs[0] = x;

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {
        x_mean = x + 0.5 * h_sq * gradlogpi(Sigma_inv, x);

        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        x = x_mean + h * x_dot;

        if ((i % thin == 0) && (i <= iter))
        {
            xs[i / thin] = x;
        }
        
        if (i == iter_reference)
        {
            x_reference = x;
        }
    }

    for (int i = 0; i < d; i++)
    {
        y(i) = RNG::rnorm(RNG::rng);
    }
    y = disp * y;

    // Coupled iterations
    int i = L; // Iteration number tracker

    while(tau == -1) 
    {
        w2_bound_parts.push_back((x - y).squaredNorm());
        i = i + 1;

        x_mean = x + 0.5 * h_sq * gradlogpi(Sigma_inv, x);
        
        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        x = x_mean + h * x_dot;

        // Reflection-maximal coupling for Y
        y_mean = y + 0.5 * h_sq * gradlogpi(Sigma_inv, y);
        z = (x_mean - y_mean) / h;

        logcpl = -0.5 * (x_dot + z).squaredNorm() + 0.5 * x_dot.squaredNorm();
        ucpl = RNG::runif(RNG::rng);

        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling; exit afterwards
        {
            tau = i;
        }
        else // Reflect
        {
            z_sqnorm = z.squaredNorm();
            y = y_mean + h * (x_dot - 2 / z_sqnorm * z.dot(x_dot) * z); // y_dot is x_dot relflected in hyperplane perpendicular to z and passing through origin
        }

        // Store
        if ((i % thin == 0) && (i <= iter))
        {
            xs[i / thin] = x;
        }

        if (i == iter_reference)
        {
            x_reference = x;   
        }
    }

    // Final iterations, after coupling, if necessary
    if (tau <= iter_reference)
    {
        for (int i = tau + 1; i <= iter_reference; i++)
        {
            x_mean = x + 0.5 * h_sq * gradlogpi(Sigma_inv, x);

            // Proposal noise
            for (int j = 0; j < d; j++)
            {
                x_dot(j) = RNG::rnorm(RNG::rng);
            }
            x = x_mean + h * x_dot;

            if ((i % thin == 0) && (i <= iter))
            {
                xs[i / thin] = x;
            }

            if (i == iter_reference)
            {
                x_reference = x;   
            }
        }
    }

    return Rcpp::List::create(Rcpp::Named("x")              = Rcpp::wrap(xs),
                              Rcpp::Named("x_reference")    = x_reference,
                              Rcpp::Named("tau")            = tau,
                              Rcpp::Named("w2_bound_parts") = Rcpp::NumericVector(w2_bound_parts.begin(), w2_bound_parts.end()));
}

//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateReflMaxMALA_scaling(const Eigen::Map<Eigen::SparseMatrix<double>> &Sigma_inv,
                             const Eigen::Map<Eigen::SparseMatrix<double>> &U,
                             const double &disp,
                             const double &h,
                             const int &L,
                             const int &iter, 
                             const int &iter_reference,
                             const int &thin)
{
    // Declare constants
    double h_sq = h * h;
    double inv_h_sq = 1 / h_sq;
    int d = Sigma_inv.rows();

    // Declare temporary variables
    Eigen::VectorXd x(d), y(d);
    Eigen::VectorXd xp(d), yp(d);
    Eigen::VectorXd x_mean(d), y_mean(d);
    Eigen::VectorXd xp_mean(d), yp_mean(d);
    Eigen::VectorXd x_dot(d), y_dot(d);
    Eigen::VectorXd z(d);
    
    double logpi_x, logpi_y;
    double logpi_xp, logpi_yp;
    double logq_xp_x, logq_x_xp;
    double logq_yp_y, logq_y_yp;
    double logHR_x, logHR_y;
    double log_u;

    bool acc_x, acc_y;
    double ucpl, logcpl;
    bool coupled;

    double z_sqnorm;
    
    // Intialize storage
    std::vector<Eigen::VectorXd> xs(iter / thin + 1); // C++ integer division used.
    Eigen::VectorXd x_reference(d);
    int tau = -1;
    std::vector<double> w2_bound_parts; // Will be of length tau.

    // Initial state pi_0 = N(0, disp^2 I)
    for (int i = 0; i < d; i++)
    {
        x(i) = RNG::rnorm(RNG::rng);
    }
    x = disp * x;
    logpi_x = logpi(U, x);
    x_mean = x + 0.5 * h_sq * gradlogpi(Sigma_inv, x);

    xs[0] = x;

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {   
    /* Generate proposal */

        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

    // Get Hastings ratio //

        logpi_xp = logpi(U, xp);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

    // Perform acceptance step //

        log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        }

    // Store values for empirical bound //

        if ((i % thin == 0) && (i <= iter))
        {
            xs[i / thin] = x;
        }

        if (i == iter_reference)
        {
            x_reference = x;
        }
    }

    for (int i = 0; i < d; i++)
    {
        y(i) = RNG::rnorm(RNG::rng);
    }
    y = disp * y;
    logpi_y = logpi(U, y);
    y_mean = y + 0.5 * h_sq * gradlogpi(Sigma_inv, y);

    // Coupled iterations
    int i = L; // Iteration number tracker

    while(tau == -1) 
    {
        w2_bound_parts.push_back((x - y).squaredNorm()); // This needs to be the squared norm. Will be of length tau - L, its last entry should be non-zero.

        i = i + 1;
        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;
        xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

        // Compute log of Hastings ratio for X
        logpi_xp = logpi(U, xp);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp
        
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Reflection-maximal coupling for Y
        z = (x_mean - y_mean) / h;

        logcpl = -0.5 * (x_dot + z).squaredNorm() + 0.5 * x_dot.squaredNorm();
        ucpl = RNG::runif(RNG::rng);

        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling of proposals
        {
            coupled = true;

            yp = xp;
            yp_mean = xp_mean;

            logpi_yp = logpi_xp;
            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).squaredNorm(); // Go from yp = xp to y
            logq_y_yp = -0.5 * inv_h_sq * (yp - y_mean).squaredNorm(); // Go from y to yp = xp

        }
        else // Reflect
        {
            coupled = false;

            z_sqnorm = z.squaredNorm();

            yp = y_mean + h * (x_dot - 2 / z_sqnorm * z.dot(x_dot) * z);
            yp_mean = yp + 0.5 * h_sq * gradlogpi(Sigma_inv, yp);

            logpi_yp = logpi(U, yp);
            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).squaredNorm(); // Go from yp to y
            logq_y_yp = logq_x_xp;                                     // Go from y to yp

        }

        // Compute log of Hastings ratio for Y
        logHR_y = logpi_yp + logq_yp_y - logpi_y - logq_y_yp;

        // Accept-reject with CRN for X,Y
        log_u = log(RNG::runif(RNG::rng));

        acc_x = false; acc_y = false;
        if (logHR_x > 0 || log_u < logHR_x)
        {
            acc_x = true;
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        } 

        if (logHR_y > 0 || log_u < logHR_y)
        {
            acc_y = true;
            y = yp;
            logpi_y = logpi_yp;
            y_mean = yp_mean;
        }

        // Store
        if ((i % thin == 0) && (i <= iter))
        {
            xs[i / thin] = x;
        }

        // Stop if coupled
        if (acc_x * acc_y * coupled == true)
        {
            tau = i;
        } 
        if (i == iter_reference)
        {
            x_reference = x;
        }
    }

    // Final iterations, after coupling
    if (tau <= iter_reference)
    {
        for (int i = tau + 1; i <= iter_reference; i++)
        {
            // Proposal noise
            for (int j = 0; j < d; j++)
            {
                x_dot(j) = RNG::rnorm(RNG::rng);
            }
            xp = x_mean + h * x_dot;

            logpi_xp = logpi(U, xp);
            xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

            // Proposal log-densities
            logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
            logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp

            // Compute log of Hastings ratio
            logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

            log_u = log(RNG::runif(RNG::rng));
            if (logHR_x > 0 || log_u < logHR_x)
            {
                x = xp;
                logpi_x = logpi_xp;
                x_mean = xp_mean;
            }

            if ((i % thin == 0) && (i <= iter))
            {
                xs[i / thin] = x;
            }
            if (i == iter_reference)
            {
                x_reference = x;
            }
        }
    }
    return Rcpp::List::create(Rcpp::Named("x")              = Rcpp::wrap(xs),
                              Rcpp::Named("x_reference")    = x_reference,
                              Rcpp::Named("tau")            = tau,
                              Rcpp::Named("w2_bound_parts") = Rcpp::NumericVector(w2_bound_parts.begin(), w2_bound_parts.end()));
}



/* OLD CODE


//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List SimulateReflMaxMALA_scaling(const Eigen::Map<Eigen::SparseMatrix<double>> &Sigma_inv,
                             const Eigen::Map<Eigen::SparseMatrix<double>> &U,
                             const double &disp,
                             const double &h,
                             const int &L,
                             const int &iter, 
                             const int &iter_reference,
                             const int &thin)
{
    // Declare constants
    double h_sq = h * h;
    double inv_h_sq = 1 / h_sq;
    int d = Sigma_inv.rows();

    // Declare temporary variables
    Eigen::VectorXd x(d), y(d);
    Eigen::VectorXd xp(d), yp(d);
    Eigen::VectorXd x_mean(d), y_mean(d);
    Eigen::VectorXd xp_mean(d), yp_mean(d);
    Eigen::VectorXd x_dot(d), y_dot(d);
    Eigen::VectorXd z(d);
    
    double logpi_x, logpi_y;
    double logpi_xp, logpi_yp;
    double logq_xp_x, logq_x_xp;
    double logq_yp_y, logq_y_yp;
    double logHR_x, logHR_y;
    double log_u;

    bool acc_x, acc_y;
    double ucpl, logcpl;

    double z_sqnorm;
    
    // Intialize storage
    std::vector<Eigen::VectorXd> xs(iter / thin + 1); // C++ integer division used.
    Eigen::VectorXd x_reference(d);
    int tau = -1;
    std::vector<double> w2_bound_parts; // Will be of length tau.

    // Initial state pi_0 = N(0, disp^2 I)
    for (int i = 0; i < d; i++)
    {
        x(i) = RNG::rnorm(RNG::rng);
    }
    x = disp * x;
    logpi_x = logpi(U, x);
    x_mean = x + 0.5 * h_sq * gradlogpi(Sigma_inv, x);

    xs[0] = x;

    // L iterations of X-chain
    for (int i = 1; i <= L; i++)
    {   
    // Generate proposal //

        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;

    // Get Hastings ratio //

        logpi_xp = logpi(U, xp);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

    // Perform acceptance step //

        log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            x = xp;
            logpi_x = logpi_xp;
            x_mean = xp_mean;
        }

    // Store values for empirical bound //

        if ((i % thin == 0) && (i <= iter))
        {
            xs[i / thin] = x;
        }

        if (i == iter_reference)
        {
            x_reference = x;
        }
    }

    for (int i = 0; i < d; i++)
    {
        y(i) = RNG::rnorm(RNG::rng);
    }
    y = disp * y;
    logpi_y = logpi(U, y);
    y_mean = y + 0.5 * h_sq * gradlogpi(Sigma_inv, y);

    // Coupled iterations
    int i = L; // Iteration number tracker

    while(tau == -1) 
    {
        w2_bound_parts.push_back((x - y).squaredNorm()); // This needs to be the squared norm. Will be of length tau - L, its last entry should be non-zero.

        i = i + 1;
        // Proposal noise
        for (int j = 0; j < d; j++)
        {
            x_dot(j) = RNG::rnorm(RNG::rng);
        }
        xp = x_mean + h * x_dot;

        logpi_xp = logpi(U, xp);
        xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

        // Proposal log-densities
        logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
        logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp

        // Compute log of Hastings ratio
        logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

        // Reflection-maximal coupling for Y
        z = (x_mean - y_mean) / h;

        logcpl = -0.5 * (x_dot + z).squaredNorm() + 0.5 * x_dot.squaredNorm();
        ucpl = RNG::runif(RNG::rng);

        log_u = log(RNG::runif(RNG::rng));
        if (logcpl > 0 || log(ucpl) < logcpl) // Maximal coupling of proposals
        {
            logq_yp_y = -0.5 * inv_h_sq * (y - xp_mean).squaredNorm(); // Go from yp = xp to y
            logq_y_yp = -0.5 * inv_h_sq * (xp - y_mean).squaredNorm(); // Go from y to yp = xp

            logHR_y = logpi_yp + logq_yp_y - logpi_y - logq_y_yp;

            // Accept-reject
            if (logHR_x > 0 || log_u < logHR_x)
            {
                acc_x = true;
                x = xp;
                logpi_x = logpi_xp;
                x_mean = xp_mean;
            } 
            else
            {
                acc_x = false;
            }

            if (logHR_y > 0 || log_u < logHR_y)
            {
                acc_y = true;
                y = xp;
                logpi_y = logpi_xp;
                y_mean = xp_mean;
            }
            else
            {
                acc_y = false;
            }

            // Store
            if ((i % thin == 0) && (i <= iter))
            {
                xs[i / thin] = x;
            }

            // Stop if coupled
            if (acc_x * acc_y == true)
            {
                tau = i;
            } 
        }
        else // Reflect
        {
            z_sqnorm = z.squaredNorm();
            yp = y_mean + h * (x_dot - 2 / z_sqnorm * z.dot(x_dot) * z);
            logpi_yp = logpi(U, yp);
            yp_mean = yp + 0.5 * h_sq * gradlogpi(Sigma_inv, yp);

            logq_yp_y = -0.5 * inv_h_sq * (y - yp_mean).squaredNorm(); // Go from yp to y
            logq_y_yp = logq_x_xp;                                     // Go from y to yp

            logHR_y = logpi_yp + logq_yp_y - logpi_y - logq_y_yp;

            // Accept-reject
            if (logHR_x > 0 || log_u < logHR_x)
            {
                x = xp;
                logpi_x = logpi_xp;
                x_mean = xp_mean;
            }
            if (logHR_y > 0 || log_u < logHR_y)
            {
                y = yp;
                logpi_y = logpi_yp;
                y_mean = yp_mean;
            }

            // Store
            if ((i % thin == 0) && (i <= iter))
            {
                xs[i / thin] = x;
            }
        }
        if (i == iter_reference)
        {
            x_reference = x;
        }
    }

    // Final iterations, after coupling
    if (tau <= iter_reference)
    {
        for (int i = tau + 1; i <= iter_reference; i++)
        {
            // Proposal noise
            for (int j = 0; j < d; j++)
            {
                x_dot(j) = RNG::rnorm(RNG::rng);
            }
            xp = x_mean + h * x_dot;

            logpi_xp = logpi(U, xp);
            xp_mean = xp + 0.5 * h_sq * gradlogpi(Sigma_inv, xp);

            // Proposal log-densities
            logq_xp_x = -0.5 * inv_h_sq * (x - xp_mean).squaredNorm(); // Go from xp to x
            logq_x_xp = -0.5 * x_dot.squaredNorm();                    // Go from x to xp

            // Compute log of Hastings ratio
            logHR_x = logpi_xp + logq_xp_x - logpi_x - logq_x_xp;

            log_u = log(RNG::runif(RNG::rng));
            if (logHR_x > 0 || log_u < logHR_x)
            {
                x = xp;
                logpi_x = logpi_xp;
                x_mean = xp_mean;
            }

            if ((i % thin == 0) && (i <= iter))
            {
                xs[i / thin] = x;
            }
            if (i == iter_reference)
            {
                x_reference = x;
            }
        }
    }
    return Rcpp::List::create(Rcpp::Named("x")              = Rcpp::wrap(xs),
                              Rcpp::Named("x_reference")    = x_reference,
                              Rcpp::Named("tau")            = tau,
                              Rcpp::Named("w2_bound_parts") = Rcpp::NumericVector(w2_bound_parts.begin(), w2_bound_parts.end()));
}
*/