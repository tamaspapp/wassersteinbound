// This file contains functions used to compute the squared Euclidean transportation cost (equivalently, W_2^2) between Gaussians, in various scenarios.

#include <RcppEigen.h>
#include <omp.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11, openmp)]]

using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::SelfAdjointEigenSolver;
using Eigen::SparseMatrix;
using Eigen::VectorXd;

using std::vector;

// Evaluate W_2^2 (w/ Euclidean norm) between two multivariate Gaussians N(a, A) and N(b, B). 
//
//' @export
// [[Rcpp::export(rng = false)]]
double EvaluateW2sq(const Eigen::Map<Eigen::VectorXd> &a, 
                    const Eigen::Map<Eigen::MatrixXd> &A, 
                    const Eigen::Map<Eigen::VectorXd> &b, 
                    const Eigen::Map<Eigen::MatrixXd> &B)
{
    SelfAdjointEigenSolver<MatrixXd> SolverA(A);
    MatrixXd A_sqrt = SolverA.operatorSqrt();

    SelfAdjointEigenSolver<MatrixXd> SolverS(A_sqrt * B * A_sqrt);

    return (a - b).squaredNorm() + A.trace() + B.trace() - 2 * SolverS.eigenvalues().array().sqrt().sum();
}

// Evaluate W_2^2 (w/ Euclidean norm) between two multivariate Gaussians N(0, A) and N(0, B). 
// For internal use only
inline double EvaluateCenteredW2sq(const Eigen::Ref<const Eigen::MatrixXd> &A, 
                                   const Eigen::Ref<const Eigen::MatrixXd> &B)
{
    SelfAdjointEigenSolver<MatrixXd> SolverA(A);
    MatrixXd A_sqrt = SolverA.operatorSqrt();

    SelfAdjointEigenSolver<MatrixXd> SolverS(A_sqrt * B * A_sqrt);

    return A.trace() + B.trace() - 2 * SolverS.eigenvalues().array().sqrt().sum();
}

// Evaluate W2^2 between marginals and target for a "single-site" Gibbs sampler with all-Gaussian marginals.
// Target is N(mu, Sigma), initial distribution is N(mu_0, Sigma_0).
//
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List EvaluateW2sqGibbs(const Eigen::Map<Eigen::VectorXd> &mu_0, 
                             const Eigen::Map<Eigen::MatrixXd> &Sigma_0, 
                             const Eigen::Map<Eigen::VectorXd> &mu, 
                             const Eigen::Map<Eigen::MatrixXd> &Sigma, 
                             const int &iter, 
                             const int &thin = 1)
{   
    // Storage
    std::vector<double> w2_sq;

    // Precompute quantities used in the Gibbs update
    MatrixXd Q = Sigma.inverse();
    int d = Q.rows();

    ArrayXd Q_diaginv = Q.diagonal().array().inverse(); // Vector of inverses of diagonal entries of Q
    MatrixXd A = MatrixXd::Identity(d, d) - Q_diaginv.matrix().asDiagonal() * Q;

    MatrixXd L = A.triangularView<Lower>();
    MatrixXd U = A - L;

    MatrixXd B = (MatrixXd::Identity(d, d) - L).inverse() * U;
    VectorXd b = mu - B * mu;

    MatrixXd add_Sigma = Sigma - B * Sigma * B.transpose();

    // Serial computation of W2^2, store with thinning

    // Evaluate and store W2^2 at first iteration
    VectorXd mean = mu_0;    // Mean and covariance of current marginal
    MatrixXd disp = Sigma_0; //

    double w2sq_now = (mean - mu).squaredNorm() + EvaluateCenteredW2sq(disp, Sigma);
    w2_sq.push_back(w2sq_now);

    for (int i = 1; i < iter + 1; i++)
    {
        mean = B * mean + b;                         // Eigen avoids aliasing here
        disp = B * disp * B.transpose() + add_Sigma; //

        if (i % thin == 0)
        {   
            w2sq_now = (mean - mu).squaredNorm() + EvaluateCenteredW2sq(disp, Sigma);
            w2_sq.push_back(w2sq_now);
        }
    }

    // Return W2^2, as well as the last marginal's mean and covariance
    return Rcpp::List::create(Rcpp::Named("w2sq")             = w2_sq, 
                              Rcpp::Named("mean_final")       = mean, 
                              Rcpp::Named("covariance_final") = disp);
}

// Compute marginals, and evaluate W2^2 between marginals and target, for a "single-site" Gibbs sampler with all-Gaussian marginals.
// Target is N(mu, Sigma), initial distribution is N(mu_0, Sigma_0).
//
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List EvaluateW2sqGibbsFull(const Eigen::Map<Eigen::VectorXd> &mu_0, 
                                 const Eigen::Map<Eigen::MatrixXd> &Sigma_0, 
                                 const Eigen::Map<Eigen::VectorXd> &mu, 
                                 const Eigen::Map<Eigen::MatrixXd> &Sigma, 
                                 const int &iter, const int &thin = 1, 
                                 const int &nthreads = 1)
{   
    // Precompute quantities
    MatrixXd Q = Sigma.inverse();
    int d = Q.rows();

    ArrayXd Q_diaginv = Q.diagonal().array().inverse(); // Vector of inverses of diagonal entries of Q
    MatrixXd A = MatrixXd::Identity(d, d) - Q_diaginv.matrix().asDiagonal() * Q;

    MatrixXd L = A.triangularView<Lower>();
    MatrixXd U = A - L;

    MatrixXd B = (MatrixXd::Identity(d, d) - L).inverse() * U;
    VectorXd b = mu - B * mu;

    MatrixXd add_Sigma = Sigma - B * Sigma * B.transpose();

    // Initialize
    VectorXd mean = mu_0;
    MatrixXd disp = Sigma_0;

    int iter_thin = iter/thin + 1; // Used C++ integer division, implicit floor in fraction

    vector<VectorXd> means(iter_thin);
    vector<MatrixXd> disps(iter_thin);

    // Perform serial computation of marginals, store with thinning
    means[0] = mean;
    disps[0] = disp;

    for (int i = 1; i < iter + 1; i++)
    {
        mean = B * mean + b;
        disp = B * disp * B.transpose() + add_Sigma;

        if (i % thin == 0)
        {
            means[i / thin] = mean;
            disps[i / thin] = disp;
        }
    }

    // Evaluate W2^2 between (thinned) marginals and target, in parallel
    ArrayXd w2_sq(iter_thin);

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for (int i = 0; i < iter_thin; i++) 
        {
            w2_sq(i) = (means[i] - mu).squaredNorm() + EvaluateCenteredW2sq(disps[i], Sigma);
        }
    }

   return Rcpp::List::create(Rcpp::Named("w2sq")        = w2_sq,
                             Rcpp::Named("means")       = means, 
                             Rcpp::Named("covariances") = disps);
}

// Evaluate W2^2 between marginals and target for ULA chain with Gaussian start and target.
//  - Target is N(mu, Sigma), and is NOT the stationary distribution. 
//  - Initial distribution is N(mu_0, Sigma_0).
//  - Step size is h (noise term multiplied by h).
//
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List EvaluateW2sqULA(const Eigen::Map<Eigen::VectorXd>             &mu_0,
                           const Eigen::Map<Eigen::SparseMatrix<double>> &Sigma_0,
                           const Eigen::Map<Eigen::VectorXd>             &mu,
                           const Eigen::Map<Eigen::MatrixXd>             &Sigma_ULA,   // Feed in from R: Sigma_ULA = (I - h^2 / 4 * Sigma_inv)^{-1} * Sigma
                           const Eigen::Map<Eigen::SparseMatrix<double>> &M,           // Feed in from R: M = I - h^2 / 2 * Sigma_inv. Sigma_inv is sparse in the simulation associated to the code.
                           const Rcpp::IntegerVector                     &which_iters, // Vector of increasing integers, storing which iterations the squared distance is to be evaluated at
                           const double &h)
{
    // Constants
    int iter = Rcpp::max(which_iters);
    int length = which_iters.size();
    double h_sq = h * h;

    // Mean and covariance at each iteration
    VectorXd mean_diff = mu_0 - mu; // Work with mean difference (\mu_t - \mu_infty) for convenience
    MatrixXd disp = Sigma_0;

    // Storage
    VectorXd w2_sq(length);

    int j = 0; // Iterator, keeps track of how many w2_sq's have been computed
    
    // Compute W2^2 between marginal and target
    if (0 == which_iters(j))
    {
        // Compute W2^2 between initial and target
        w2_sq(j) = mean_diff.squaredNorm() + EvaluateCenteredW2sq(disp, Sigma_ULA);
        j +=1;
    }

    // Update ULA and compute W2^2
    for (int i = 1; i <= iter; i++)
    {
        // Update mean difference
        mean_diff = M * mean_diff;

        // Update covariance
        disp = M * disp * M;
        disp.diagonal().array() += h_sq;

        // Compute W2^2 between marginal and target
        if (i == which_iters(j))
        {
            w2_sq(j) = mean_diff.squaredNorm() + EvaluateCenteredW2sq(disp, Sigma_ULA);
            j +=1;
        }
    }
    VectorXd mean = mean_diff + mu;
    return Rcpp::List::create(Rcpp::Named("w2sq")             = w2_sq, 
                              Rcpp::Named("mean_final")       = mean, 
                              Rcpp::Named("covariance_final") = disp);
}