#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

#include "samplers.h"

using Eigen::ArrayXd;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using std::string;
typedef Eigen::SparseMatrix<double> SparseMat;

/******************** Helpers ********************/

VectorXd thinned_squared_distance(const Ref<const MatrixXd> &xs, const Ref<const MatrixXd> &ys, const int &thin)
{
    return (xs(Eigen::seq(0, Eigen::last, thin), Eigen::all) - ys(Eigen::seq(0, Eigen::last, thin), Eigen::all)).rowwise().squaredNorm();
}

/******************** Single-chain MCMC ********************/

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List rwm_cpp(const Rcpp::List &target_params,
                   const Rcpp::List &sampler_params,
                   const Eigen::Map<Eigen::VectorXd> &theta0,
                   const int &iter,
                   const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);  // Proposal covariance. Can be fed in as a vector for diagonal covariances.

    // Output
    MatrixXd xs(iter / thin + 1, theta0.size());
    double acc_rate_x;

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        rwm(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        rwm(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "multivariate_logistic")
    {
        MultivariateLogistic pi;

        rwm(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        rwm(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "sparse_gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu, Omega);

        rwm(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "1d_gaussian_mixture")
    {
        ArrayXd p = Rcpp::as<Map<ArrayXd>>(target_params["p"]);
        ArrayXd mu = Rcpp::as<Map<ArrayXd>>(target_params["mu"]);
        ArrayXd sigma = Rcpp::as<Map<ArrayXd>>(target_params["sigma"]);
        OneDimGaussMixture pi(p, mu, sigma);

        rwm(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("acc_rate_x") = acc_rate_x);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List mala_cpp(const Rcpp::List &target_params,
                    const Rcpp::List &sampler_params,
                    const Eigen::Map<Eigen::VectorXd> &theta0,
                    const int &iter,
                    const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]); // Proposal covariance. Can be fed in as a vector for diagonal covariances.
    // Output
    MatrixXd xs(iter / thin + 1, theta0.size());
    double acc_rate_x;

    string target_type = target_params["target_type"];

    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        mala(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        mala(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "multivariate_logistic")
    {
        MultivariateLogistic pi;

        mala(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        mala(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu, Omega_target);

        mala(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else if (target_type == "1d_gaussian_mixture")
    {
        ArrayXd p = Rcpp::as<Map<ArrayXd>>(target_params["p"]);
        ArrayXd mu = Rcpp::as<Map<ArrayXd>>(target_params["mu"]);
        ArrayXd sigma = Rcpp::as<Map<ArrayXd>>(target_params["sigma"]);
        OneDimGaussMixture pi(p, mu, sigma);

        mala(pi, theta0, Sigma, iter, thin, xs, acc_rate_x);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("acc_rate_x") = acc_rate_x);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List ula_cpp(const Rcpp::List &target_params,
                   const Rcpp::List &sampler_params, 
                   const Eigen::Map<Eigen::VectorXd> &theta0,
                   const int &iter,
                   const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]); // Proposal covariance. Can be fed in as a vector for diagonal covariances.
    // Output
    MatrixXd xs(iter / thin + 1, theta0.size());

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        ula(pi, theta0, Sigma, iter, thin, xs);
    }
    else if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        ula(pi, theta0, Sigma, iter, thin, xs);
    }
    else if (target_type == "multivariate_logistic")
    {
        MultivariateLogistic pi;

        ula(pi, theta0, Sigma, iter, thin, xs);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        ula(pi, theta0, Sigma, iter, thin, xs);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu, Omega_target);

        ula(pi, theta0, Sigma, iter, thin, xs);
    }
    else if (target_type == "1d_gaussian_mixture")
    {
        ArrayXd p = Rcpp::as<Map<ArrayXd>>(target_params["p"]);
        ArrayXd mu = Rcpp::as<Map<ArrayXd>>(target_params["mu"]);
        ArrayXd sigma = Rcpp::as<Map<ArrayXd>>(target_params["sigma"]);
        OneDimGaussMixture pi(p, mu, sigma);

        ula(pi, theta0, Sigma, iter, thin, xs);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List obab_cpp(const Rcpp::List &target_params,
                    const Rcpp::List &sampler_params,
                    const Eigen::Map<Eigen::VectorXd> &x0,
                    const int &iter,
                    const int &thin)

{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double gamma = sampler_params["gamma"];
    double delta = sampler_params["delta"];
    double eta = exp(-delta * gamma);

    // Output
    MatrixXd xs(iter / thin + 1, x0.size());

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        obab(pi, x0, Sigma, eta, delta, iter, thin, xs);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        obab(pi, x0, Sigma, eta, delta, iter, thin, xs);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu, Omega_target);

        obab(pi, x0, Sigma, eta, delta, iter, thin, xs);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List horowitz_cpp(const Rcpp::List &target_params,
                        const Rcpp::List &sampler_params,
                        const Eigen::Map<Eigen::VectorXd> &x0,
                        const int &iter,
                        const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double gamma = sampler_params["gamma"];
    double delta = sampler_params["delta"];
    double eta = exp(-delta * gamma);

    // Output
    MatrixXd xs(iter / thin + 1, x0.size());
    std::vector<bool> acceptances_x(iter);

    string target_type = target_params["target_type"];
    if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        horowitz(pi, x0, Sigma, eta, delta, iter, thin, xs, acceptances_x);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu_target, Omega_target);

        horowitz(pi, x0, Sigma, eta, delta, iter, thin, xs, acceptances_x);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("acceptances_x") = acceptances_x);
}

/************** Adaptive MCMC ***************/

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List fisher_mala_cpp(const Rcpp::List &target_params,
                           const Rcpp::List &sampler_params,
                           const Eigen::Map<Eigen::VectorXd> &x0,
                           const int &iter,
                           const int &thin)
{
    // Sampler parameters
    double sigma0 = sampler_params["sigma0"];
    double acc_target = sampler_params["acceptance_rate_target"];
    double lr = sampler_params["learning_rate"];
    double damping = sampler_params["damping_factor"];

    // Adaptation parameters, i.e. proposal square-root and its scalar multiplier
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(x0.size(), x0.size());
    double sigma = sigma0;

    // Output
    MatrixXd xs(iter / thin + 1, x0.size());
    std::vector<bool> acceptances_x(iter, false);

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        fisher_mala(pi, x0, sigma, R, lr, acc_target, damping, iter, thin, xs, acceptances_x);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        fisher_mala(pi, x0, sigma, R, lr, acc_target, damping, iter, thin, xs, acceptances_x);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu_target, Omega_target);

        fisher_mala(pi, x0, sigma, R, lr, acc_target, damping, iter, thin, xs, acceptances_x);
    }
    else if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        fisher_mala(pi, x0, sigma, R, lr, acc_target, damping, iter, thin, xs, acceptances_x);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("R") = R,
                              Rcpp::Named("sigma") = sigma,
                              Rcpp::Named("acceptances_x") = acceptances_x);
}

/************** Coupled MCMC, same kernel ***************/

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List rwm_twoscalegcrn_cpp(const Rcpp::List &target_params,
                                const Rcpp::List &sampler_params, // Proposal covariance. Can be fed in as a vector for diagonal covariances.
                                const Eigen::Map<Eigen::VectorXd> &x0,
                                const Eigen::Map<Eigen::VectorXd> &y0,
                                const int &iter,
                                const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double thresh = sampler_params["thresh"];
    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    double acc_rate_x, acc_rate_y;
    int tau = -1;

    string target_type = target_params["target_type"];
    if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        rwm_twoscaleGCRN(pi, x0, y0, Sigma, iter, thin, thresh, xs, ys, acc_rate_x, acc_rate_y, tau);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }
    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("tau") = tau,
                              Rcpp::Named("acc_rate_x") = acc_rate_x,
                              Rcpp::Named("acc_rate_y") = acc_rate_y);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List ula_twoscalecrn_cpp(const Rcpp::List &target_params,
                               const Rcpp::List &sampler_params, // Proposal covariance. Can be fed in as a vector for diagonal covariances.
                               const Eigen::Map<Eigen::VectorXd> &x0,
                               const Eigen::Map<Eigen::VectorXd> &y0,
                               const int &iter,
                               const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double thresh = sampler_params["thresh"];

    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    int tau = -1;

    string target_type = target_params["target_type"];

    if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        ula_twoscaleCRN(pi, x0, y0, Sigma, iter, thin, thresh, xs, ys, tau);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu, Omega_target);

        ula_twoscaleCRN(pi, x0, y0, Sigma, iter, thin, thresh, xs, ys, tau);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }
    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("tau") = tau);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List mala_twoscalecrn_cpp(const Rcpp::List &target_params,
                                const Rcpp::List &sampler_params,
                                const Eigen::Map<Eigen::VectorXd> &x0,
                                const Eigen::Map<Eigen::VectorXd> &y0,
                                const int &iter,
                                const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double thresh = sampler_params["thresh"];
    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    double acc_rate_x, acc_rate_y;
    int tau = -1;

    string target_type = target_params["target_type"];

    if (target_type == "stochastic_volatility")
    {
        Map<VectorXd> y_svm = Rcpp::as<Map<VectorXd>>(target_params["y"]);
        double beta_svm = target_params["beta"];
        double sigma_svm = target_params["sigma"];
        double phi_svm = target_params["phi"];
        StochasticVolatility pi(y_svm, beta_svm, sigma_svm, phi_svm);

        mala_twoscaleCRN(pi, x0, y0, Sigma, iter, thin, thresh, xs, ys, acc_rate_x, acc_rate_y, tau);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        mala_twoscaleCRN(pi, x0, y0, Sigma, iter, thin, thresh, xs, ys, acc_rate_x, acc_rate_y, tau);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu, Omega_target);

        mala_twoscaleCRN(pi, x0, y0, Sigma, iter, thin, thresh, xs, ys, acc_rate_x, acc_rate_y, tau);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }
    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("tau") = tau,
                              Rcpp::Named("acc_rate_x") = acc_rate_x,
                              Rcpp::Named("acc_rate_y") = acc_rate_y);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List horowitz_CRN_cpp(const Rcpp::List &target_params,
                            const Rcpp::List &sampler_params,
                            const Eigen::Map<Eigen::VectorXd> &x0,
                            const Eigen::Map<Eigen::VectorXd> &y0,
                            const int &iter,
                            const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double gamma = sampler_params["gamma"];
    double delta = sampler_params["delta"];
    double eta = exp(-delta * gamma);

    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    std::vector<bool> acceptances_x(iter), acceptances_y(iter);

    string target_type = target_params["target_type"];
    if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        horowitz_CRN(pi, x0, y0, Sigma, eta, delta, iter, thin, xs, ys, acceptances_x, acceptances_y);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu_target, Omega_target);

        horowitz_CRN(pi, x0, y0, Sigma, eta, delta, iter, thin, xs, ys, acceptances_x, acceptances_y);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("acceptances_x") = acceptances_x,
                              Rcpp::Named("acceptances_y") = acceptances_y);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List obab_CRN_cpp(const Rcpp::List &target_params,
                        const Rcpp::List &sampler_params,
                        const Eigen::Map<Eigen::VectorXd> &x0,
                        const Eigen::Map<Eigen::VectorXd> &y0,
                        const int &iter,
                        const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double gamma = sampler_params["gamma"];
    double delta = sampler_params["delta"];
    double eta = exp(-delta * gamma);

    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());

    string target_type = target_params["target_type"];
    if (target_type == "gaussian")
    {
        Map<VectorXd> mu = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu, Omega);

        obab_CRN(pi, x0, y0, Sigma, eta, delta, iter, thin, xs, ys);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu_target, Omega_target);

        obab_CRN(pi, x0, y0, Sigma, eta, delta, iter, thin, xs, ys);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys);
}

/************* Coupled MCMC, different kernels *************/

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List ula_mala_CRN_cpp(const Rcpp::List &target_params,
                            const Rcpp::List &sampler_params,
                            const Eigen::Map<Eigen::VectorXd> &x0,
                            const Eigen::Map<Eigen::VectorXd> &y0,
                            const int &iter,
                            const int &thin)
{
    // Input 
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);

    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    double acc_rate_y;

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        ula_mala_CRN(pi, x0, y0, Sigma, iter, thin, xs, ys, acc_rate_y);
    }
    else if (target_type == "gaussian")
    {
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega_target = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu_target, Omega_target);

        ula_mala_CRN(pi, x0, y0, Sigma, iter, thin, xs, ys, acc_rate_y);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu_target, Omega_target);

        ula_mala_CRN(pi, x0, y0, Sigma, iter, thin, xs, ys, acc_rate_y);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }
    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("acc_rate_y") = acc_rate_y);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List obab_horowitz_CRN_cpp(const Rcpp::List &target_params,
                                 const Rcpp::List &sampler_params,
                                 const Eigen::Map<Eigen::VectorXd> &x0,
                                 const Eigen::Map<Eigen::VectorXd> &y0,
                                 const int &iter,
                                 const int &thin)
{
    // Sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    double gamma = sampler_params["gamma"];
    double delta = sampler_params["delta"];
    double eta = exp(-delta * gamma);

    // Output
    MatrixXd xs(iter / thin + 1, x0.size());
    MatrixXd ys(iter / thin + 1, y0.size());
    std::vector<bool> acceptances_y(iter);

    string target_type = target_params["target_type"];
    if (target_type == "gaussian")
    {
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<MatrixXd> Omega_target = Rcpp::as<Map<MatrixXd>>(target_params["Omega"]);
        Gaussian pi(mu_target, Omega_target);

        obab_horowitz_CRN(pi, x0, y0, Sigma, eta, delta, iter, thin, xs, ys, acceptances_y);
    }
    else if (target_type == "sparse_gaussian")
    {
        // Careful that we don't overwrite the sampler parameters!
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_params["mu"]);
        Map<SparseMat> Omega_target = Rcpp::as<Map<SparseMat>>(target_params["Omega"]);
        SparseGaussian pi(mu_target, Omega_target);

        obab_horowitz_CRN(pi, x0, y0, Sigma, eta, delta, iter, thin, xs, ys, acceptances_y);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }
    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("acceptances_y") = acceptances_y);
}

//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List mala_CRN_2targets_cpp(const Rcpp::List &target_x_params,
                                 const Rcpp::List &target_y_params,
                                 const Rcpp::List &sampler_params,
                                 const Eigen::Map<Eigen::VectorXd> &x0,
                                 const Eigen::Map<Eigen::VectorXd> &y0,
                                 const int &iter,
                                 const int &thin)
{
    //Input
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    
    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    double acc_rate_x, acc_rate_y;

    string target_x_type = target_x_params["target_type"];
    string target_y_type = target_y_params["target_type"];

    if (target_x_type == "gaussian" && target_y_type == "logistic_regression")
    {
        Map<VectorXd> mu_target = Rcpp::as<Map<VectorXd>>(target_x_params["mu"]);
        Map<MatrixXd> Omega_target = Rcpp::as<Map<MatrixXd>>(target_x_params["Omega"]);
        Gaussian pi_x(mu_target, Omega_target);

        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_y_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_y_params["lambda"]);
        LogisticRegression pi_y(yX, lambda);

        mala_CRN_2targets(pi_x, pi_y, x0, y0, Sigma, iter, thin, xs, ys, acc_rate_x, acc_rate_y);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }
    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);

    return Rcpp::List::create(Rcpp::Named("squaredist") = squaredist,
                              Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("acc_rate_x") = acc_rate_x,
                              Rcpp::Named("acc_rate_y") = acc_rate_y);
}

/*********** Tall data MCMC samplers ************/

//' SGLD: Stochastic gradient Langevin dynamics
//'
//' At each iteration, batches of "batch_size" are sampled uniformly without replacement. The draws are independent across iterations, i.e. we don't enforce "epochs" where we cycle over the whole dataset.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List sgld_cpp(const Rcpp::List &target_params,
                    const Rcpp::List &sampler_params,
                    const Eigen::Map<Eigen::VectorXd> &theta0,
                    const int &iter,
                    const int &thin)
{
    // Unpack sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    int batch_size = sampler_params["batch_size"];

    // Output
    MatrixXd samples(iter + 1, theta0.size());

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        sgld(pi, batch_size, theta0, Sigma, iter, thin, samples);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = samples);
}

//' SGLD-CV: Stochastic gradient Langevin dynamics, with control variates.
//'
//' At each iteration, batches of "batch_size" are sampled uniformly without replacement. The draws are independent across iterations, i.e. we don't enforce "epochs" where we cycle over the whole dataset.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List sgldcv_cpp(const Rcpp::List &target_params,
                      const Rcpp::List &sampler_params,
                      const Eigen::Map<Eigen::VectorXd> &theta0,
                      const int &iter,
                      const int &thin)
{
    // Unpack sampler parameters
    Map<VectorXd> mode = Rcpp::as<Map<VectorXd>>(sampler_params["mode"]);
    Map<VectorXd> gradient_at_mode = Rcpp::as<Map<VectorXd>>(sampler_params["gradient_at_mode"]);
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    int batch_size = sampler_params["batch_size"];

    // Output
    MatrixXd samples(iter + 1, theta0.size());

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        sgldcv(pi, mode, gradient_at_mode, batch_size, theta0, Sigma, iter, thin, samples);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    return Rcpp::List::create(Rcpp::Named("xs") = samples);
}

/************* Coupled MCMC samplers *************/

//' SGLD coupled with MALA (with exact gradient), CRN coupling.
//'
//' At each iteration of SGLD, batches of "batch_size" are sampled uniformly without replacement. The draws are independent across iterations, i.e. we don't enforce "epochs" where we cycle over the whole dataset.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List sgld_mala_CRN_cpp(const Rcpp::List &target_params,
                             const Rcpp::List &sampler_params,
                             const Eigen::Map<Eigen::VectorXd> &x0,
                             const Eigen::Map<Eigen::VectorXd> &y0,
                             const int &iter,
                             const int &thin)
{
    // Unpack sampler parameters
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    int batch_size = sampler_params["batch_size"];

    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    double acc_rate_y;

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        sgld_mala_CRN(pi, batch_size, x0, y0, Sigma, iter, thin, xs, ys, acc_rate_y);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);
    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("acc_rate_y") = acc_rate_y,
                              Rcpp::Named("squaredist") = squaredist);
}

//' SGLD-CV coupled with MALA (with exact gradient), CRN coupling.
//'
//' At each iteration of SGLD, batches of "batch_size" are sampled uniformly without replacement. The draws are independent across iterations, i.e. we don't enforce "epochs" where we cycle over the whole dataset.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List sgldcv_mala_CRN_cpp(const Rcpp::List &target_params,
                               const Rcpp::List &sampler_params,
                               const Eigen::Map<Eigen::VectorXd> &x0,
                               const Eigen::Map<Eigen::VectorXd> &y0,
                               const int &iter,
                               const int &thin)
{
    // Unpack sampler parameters
    Map<VectorXd> mode = Rcpp::as<Map<VectorXd>>(sampler_params["mode"]);
    Map<VectorXd> gradient_at_mode = Rcpp::as<Map<VectorXd>>(sampler_params["gradient_at_mode"]);
    Map<MatrixXd> Sigma = Rcpp::as<Map<MatrixXd>>(sampler_params["Sigma"]);
    int batch_size = sampler_params["batch_size"];

    // Output
    MatrixXd xs(iter / thin + 1, x0.size()), ys(iter / thin + 1, y0.size());
    double acc_rate_y;

    string target_type = target_params["target_type"];
    if (target_type == "logistic_regression")
    {
        Map<MatrixXd> yX = Rcpp::as<Map<MatrixXd>>(target_params["yX"]);
        Map<VectorXd> lambda = Rcpp::as<Map<VectorXd>>(target_params["lambda"]);
        LogisticRegression pi(yX, lambda);

        sgldcv_mala_CRN(pi, mode, gradient_at_mode, batch_size, x0, y0, Sigma, iter, thin, xs, ys, acc_rate_y);
    }
    else
    {
        std::cout << "Target type not implemented."
                  << "\n";
        assert(false);
    }

    VectorXd squaredist = thinned_squared_distance(xs, ys, 1);
    return Rcpp::List::create(Rcpp::Named("xs") = xs,
                              Rcpp::Named("ys") = ys,
                              Rcpp::Named("acc_rate_y") = acc_rate_y,
                              Rcpp::Named("squaredist") = squaredist);
}
