#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using Eigen::ArrayXd;

//' Compute 1D optimal transport cost and dual solution from sorted (equal-size) vectors of samples
//'
//' The optimal dual solution is recovered by sorting the two samples, then applying  Algorithm 3 of 
//' https://proceedings.mlr.press/v151/sejourne22a/sejourne22a.pdf, which simplifies considerably in our case.
//' (There are typos in the paper: line 4 of Algorithm 3 should be "a <= b"; the initialization should be g_1 = C(x_1, y_1).
//' See https://github.com/thibsej/fast_uot/blob/main/fastuot/uot1d.py for a correct implementation.)
//' 
//' @export
// [[Rcpp::export(rng=false)]]
Rcpp::List w2sq_1d_dual_cpp(Eigen::ArrayXd x_sorted, Eigen::ArrayXd y_sorted)
{
    auto cost = [&](int i, int j) { double root = x_sorted[i] - y_sorted[j]; return root*root; };
    
    int n = x_sorted.size();

    ArrayXd x_potential = ArrayXd::Zero(n), y_potential = ArrayXd::Zero(n); 
    y_potential(0) = cost(0,0);
    for(int i = 1; i < n; i++)
    {
        x_potential(i) = cost(i,i-1) - y_potential(i-1);
        y_potential(i) = cost(i,i) - x_potential(i);

    }

    return Rcpp::List::create(Rcpp::Named("w2sq") = (x_potential + y_potential).mean(),
                              Rcpp::Named("potentials_x") = x_potential,
                              Rcpp::Named("potentials_y") = y_potential);                
}
