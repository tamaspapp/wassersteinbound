#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "cost_eval.h"

Eigen::MatrixXd squared_euclidean_cost_matrix(const Eigen::Ref<const Eigen::MatrixXd> &x,
                                              const Eigen::Ref<const Eigen::MatrixXd> &y)
{
    Eigen::VectorXd xsq    = x.rowwise().squaredNorm();
    Eigen::RowVectorXd ysq = y.rowwise().squaredNorm();
    Eigen::MatrixXd cost = ((-2.) * x) * y.transpose();
    cost.colwise() += xsq;
    cost.rowwise() += ysq;
    return cost;
}

typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> RowMajorMat;
RowMajorMat squared_euclidean_cost_matrix_rowmajor(const Eigen::Ref<const Eigen::MatrixXd> &x,
                                                   const Eigen::Ref<const Eigen::MatrixXd> &y)
{
    Eigen::VectorXd xsq = x.rowwise().squaredNorm();
    Eigen::RowVectorXd ysq = y.rowwise().squaredNorm();
    RowMajorMat cost = (-2.) * (x * y.transpose());
    cost.colwise() += xsq;
    cost.rowwise() += ysq;
    return cost;
}

// Eigen::MatrixXf squared_euclidean_cost_matrix_float(const Eigen::Ref<const Eigen::MatrixXf> &x,
//                                                     const Eigen::Ref<const Eigen::MatrixXf> &y)
// {
//     Eigen::VectorXf xsq    = x.rowwise().squaredNorm();
//     Eigen::RowVectorXf ysq = y.rowwise().squaredNorm();
//     Eigen::MatrixXf cost = ((-2.) * x) * y.transpose();
//     cost.colwise() += xsq;
//     cost.rowwise() += ysq;
//     return cost;
// }


//' Evaluate squared Euclidean cost matrix.
//'      Input is assumed to be in row-major format ("n" times "d").
//'
//' @export
//[[Rcpp::export(rng = false)]]
Eigen::MatrixXd EvaluateSquaredCost(const Eigen::Map<Eigen::MatrixXd> &x, 
                                    const Eigen::Map<Eigen::MatrixXd> &y)
{
    return squared_euclidean_cost_matrix(x, y);
}

