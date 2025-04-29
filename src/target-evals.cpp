#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

#include "targets.h"

using Eigen::ArrayXd;
using Eigen::SparseMatrix;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::string;

/************* Target evaluation **********/

//' @export
// [[Rcpp::export(rng = false)]]
double potential_cpp(const Rcpp::List &target_params, 
                     const Eigen::Map<Eigen::VectorXd> &theta)
{
    string target_type = target_params["target_type"];

    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        return pi.Potential(theta);
    }
    else if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        return -pi.LogDensity(theta);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        return 0;
    }
}

//' @export
// [[Rcpp::export(rng = false)]]
Eigen::VectorXd gradpotential_cpp(const Rcpp::List &target_params, 
                                  const Eigen::Map<Eigen::VectorXd> &theta)
{
    string target_type = target_params["target_type"];


    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        return pi.GradientPotential(theta);
    }
    else if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        return -pi.GradientLogDensity(theta);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        return Eigen::VectorXd::Zero(theta.size());
    }
}

//' @export
// [[Rcpp::export(rng = false)]]
Eigen::MatrixXd hesspotential_cpp(const Rcpp::List &target_params, 
                                  const Eigen::Map<Eigen::VectorXd> &theta)
{
    string target_type = target_params["target_type"];

    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        return pi.HessianPotential(theta);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        return Eigen::MatrixXd::Zero(theta.size(), theta.size());
    }
}
