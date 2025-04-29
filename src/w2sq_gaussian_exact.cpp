#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;
using namespace std;

// Matrix power by repeated squaring. Assumes n > 0.
inline MatrixXd matrix_power(const Ref<const MatrixXd> A, const int &n) // Could be templated
{
  if(n == 1) return A;
  
  MatrixXd A_ = A;
  int n_ = n;
  MatrixXd res = MatrixXd::Identity(A.rows(), A.cols());
    while (n_ > 0) {
        if (n_ & 1) res = res * A_; // If the smallest bit is 1, increment. 
        n_ >>= 1; // Cut off the smallest bit
        if(n_ > 0) A_ = A_ * A_;
    }
  return res;
}

//' @export
// [[Rcpp::export(rng = false)]]
Eigen::MatrixXd matrix_power_cpp(const Eigen::Map<Eigen::MatrixXd> A, const int &n) {
    return matrix_power(A, n);
}

// tr(sqrt(sqrt(A) * B * sqrt(A)))
// Assumes that B is symmetric and that sqrtA is symmetric square root of A (i.e. the "principal square root")
inline double trace_sqrtof_sqrtA_B_sqrtA(const Ref<const MatrixXd> &sqrtA, 
                                         const Ref<const MatrixXd> &B) 
{
    SelfAdjointEigenSolver<MatrixXd> SolverS(sqrtA * B * sqrtA);
    double rv = SolverS.eigenvalues().cwiseSqrt().sum();
    return rv;
}

//' Evaluate W_2^2 (w/ Euclidean norm) between two multivariate Gaussians N(a, A) and N(b, B). 
//'
//' @export
// [[Rcpp::export(rng = false)]]
double w2sq_gaussian_cpp(const Eigen::Map<Eigen::VectorXd> &a, const Eigen::Map<Eigen::MatrixXd> &A, 
                         const Eigen::Map<Eigen::VectorXd> &b, const Eigen::Map<Eigen::MatrixXd> &B)
{
    SelfAdjointEigenSolver<MatrixXd> SolverA(A);
    MatrixXd sqrtA = SolverA.operatorSqrt();

    return (a - b).squaredNorm() + A.trace() + B.trace() - 2. * trace_sqrtof_sqrtA_B_sqrtA(sqrtA, B);
}


//' Evaluate W_2^2 (w/ Euclidean norm) between first d-dimensions of marginal N(mu_t[1:d], Sigma_t[1:d]) and target N(mu[1:d], Sigma[1:d])
//' of a process obeying the autoregression:
//'              (mu_{t + 1} - mu_inf) = B (mu_t - mu_inf)
//'              (Sigma_{t + 1} - Sigma_inf) = B (Sigma_t - Sigma_inf) B.transpose()
//'
//' Any vector autoregressive process with "slope" B satisfies this recurrence relation as soon as the spectrum of B is inside the ball of radius 1.
//'
//' Deterministic Gibbs, ULA and OBAB all fall under this category.
//'
//' @export
// [[Rcpp::export(rng = false)]]                 
Rcpp::List w2sq_convergence_gaussian_recursive_cpp(const Eigen::Map<Eigen::VectorXd> &mu_0, const Eigen::Map<Eigen::MatrixXd> &Sigma_0, // Starting distribution
                                                   const Eigen::Map<Eigen::VectorXd> &mu_inf, const Eigen::Map<Eigen::MatrixXd> &Sigma_inf, // Stationary distribution
                                                   const Eigen::Map<Eigen::MatrixXd> &B, // Matrix appearing in recursion. 
                                                   const int &d, // Wasserstein distance only computed for the first d coordinates
                                                   const int &iter, const int &thin, const double &tol = 1e-12)
{
    int iter_thin = iter / thin;
    Eigen::VectorXd w2sq = Eigen::VectorXd::Zero(iter_thin + 1);

    double trSigma_inf = Sigma_inf.topLeftCorner(d, d).trace();
    SelfAdjointEigenSolver<MatrixXd> SolverSigma_inf(Sigma_inf.topLeftCorner(d, d));
    MatrixXd sqrtSigma_inf = SolverSigma_inf.operatorSqrt();

    VectorXd mu_diff = mu_0 - mu_inf;
    MatrixXd Sigma_diff = Sigma_0 - Sigma_inf;
    
    MatrixXd Sigma = Sigma_0;
    w2sq(0) = mu_diff.head(d).squaredNorm() + trSigma_inf + Sigma.topLeftCorner(d, d).trace() - 2. * trace_sqrtof_sqrtA_B_sqrtA(sqrtSigma_inf, Sigma.topLeftCorner(d, d));

    MatrixXd Bthin = matrix_power(B, thin);
    for (int i = 1; i <= iter_thin; i++) // Four matmuls and one spd matrix square-root per iteration
    {   
        mu_diff = Bthin * mu_diff; // Eigen takes care of aliasing with matmuls
        Sigma_diff = Bthin * Sigma_diff * Bthin.transpose();

        Sigma = Sigma_inf + Sigma_diff;
             
        w2sq(i) = mu_diff.head(d).squaredNorm() + trSigma_inf + Sigma.topLeftCorner(d, d).trace() - 2. * trace_sqrtof_sqrtA_B_sqrtA(sqrtSigma_inf, Sigma.topLeftCorner(d, d));
        
        // Stop if the squared distance is essentially zero
        if(w2sq(i) < tol) { w2sq(i) = 0.; break; }
    }

    return Rcpp::List::create(Rcpp::Named("w2sq") = w2sq, 
                              Rcpp::Named("Sigma_t") = Sigma,
                              Rcpp::Named("mu_t") = mu_inf + mu_diff);
}
