#include <RcppEigen.h>
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/gamma.hpp>

/***** Incomplete gamma function and its inverse: vectorize over the argument *****/

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector low_inc_gamma_cpp(const double &rate, const Rcpp::NumericVector &upper_truncation) {
    Rcpp::NumericVector out(upper_truncation.size());
    for (int i = 0; i < upper_truncation.size(); i++) out[i] =  boost::math::gamma_p(rate, upper_truncation[i]);
    return out;
} // Incomplete gamma function

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector low_inc_gamma_inv_cpp(const double &rate, const Rcpp::NumericVector &p) {
    Rcpp::NumericVector out(p.size());
    for (int i = 0; i < p.size(); i++) out[i] =  boost::math::gamma_p_inv(rate, p[i]);
    return out;
} // Inverse of incomplete gamma function


/***** Fast linear algebra *****/

//' @export
// [[Rcpp::export(rng = false)]]
Eigen::MatrixXd cpp_prod(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y)
{
    return X*Y;
} // Matrix product

//' @export
// [[Rcpp::export(rng = false)]]
Eigen::MatrixXd cpp_crossprod(const Eigen::Map<Eigen::MatrixXd> &X)
{
    return X.transpose()*X;
} // Cross product

//' @export
// [[Rcpp::export(rng = false)]]
Eigen::MatrixXd cpp_cov(const Eigen::Map<Eigen::MatrixXd> &X)
{
    Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = (X_centered.transpose() * X_centered) / double(X.rows() - 1.);
    return cov;
} // Sample covariance matrix
