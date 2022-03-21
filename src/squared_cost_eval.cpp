/*
This file contains functions used to compute the squared Euclidean cost matrix between two empirical distributions.
    - Empirical distributions are input as matrices.
    - Within each empirical distribution, each sample is either a column vector of size #{dimensions}, or a row vector of size #{dimensions}.
    - Parallel evaluation is supported via OpenMP.
*/

#include <RcppEigen.h>
#include <omp.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11, openmp)]]

// Version that takes the input in column-major order: #{dimensions}-by-#{samples}.
//
//' @export
//[[Rcpp::export(rng = false)]]
Eigen::ArrayXXd EvaluateSquaredCost(const Eigen::Map<Eigen::MatrixXd> &x, const Eigen::Map<Eigen::MatrixXd> &y, const int &nthreads = 1) 
{
    Eigen::ArrayXXd cost(x.cols(), y.cols());

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for (int i = 0; i < y.cols(); i++)
        {
            cost.col(i) = (x.colwise() - y.col(i)).colwise().squaredNorm();
        }
    }
    
    return cost;
}

// Version that takes the input in row-major order: #{samples}-by-#{dimensions}.
// Slower than using column-major order, due to Eigen internals.
Eigen::ArrayXXd EvaluateSquaredCost_row(const Eigen::Map<Eigen::MatrixXd> &x, const Eigen::Map<Eigen::MatrixXd> &y, const int &nthreads = 1)
{
    Eigen::ArrayXXd cost(x.rows(), y.rows());

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for (int i = 0; i < y.rows(); i++)
        {
            cost.col(i) = (x.rowwise() - y.row(i)).rowwise().squaredNorm();
        }
    }
    
    return cost;
}
