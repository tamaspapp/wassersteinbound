#include <Rcpp.h>

// Compute the one-dimensional squared-Euclidean transportation cost, together with leave-one-out transportation costs used to compute a jackknife error estimate
//
// input:
// x_sorted    = Order statistics of 1d sample x. (Increasing order is assumed, decreasing order should also work.)
// inv_order_x = The inverse of the permutation which had sorted sample x. I.e. the location of the sample X_0, then X_1, and so on.
//               Indexing must start from 0!
//
//' @export
//[[Rcpp::export]]
Rcpp::List EvaluateJackknife1d (Rcpp::NumericVector x_sorted,    Rcpp::NumericVector y_sorted, 
                                Rcpp::IntegerVector inv_order_x, Rcpp::IntegerVector inv_order_y)
{
    int n = x_sorted.size();
    double cost = 0.;
    Rcpp::NumericVector jack(n, 0.);

    for (int i = 0; i < n; i++)
    {
        double d = x_sorted[i] - y_sorted[i];
        cost += d*d;
    }

    int min, max;
    for (int i = 0; i < n; i++)
    {
        if (inv_order_x[i] > inv_order_y[i])
        {
            min = inv_order_y[i];
            max = inv_order_x[i];
            
            // Skip "min" in y, then skip "max" in x
            for (int j = 0; j < min; j++)
            {
                double d = x_sorted[j] - y_sorted[j];
                jack[i] += d*d;
            }
            for (int j = min; j < max; j++)
            {
                double d = x_sorted[j] - y_sorted[j + 1];
                jack[i] += d*d;
            }
            for (int j = max + 1; j < n; j++)
            {
                double d = x_sorted[j] - y_sorted[j];
                jack[i] += d*d;
            }
        }
        else
        {
            min = inv_order_x[i];
            max = inv_order_y[i];
            
            for (int j = 0; j < min; j++)
            {
                double d = x_sorted[j] - y_sorted[j];
                jack[i] += d*d;
            }
            for (int j = min; j < max; j++)
            {
                double d = x_sorted[j + 1] - y_sorted[j];
                jack[i] += d*d;
            }
            for (int j = max + 1; j < n; j++)
            {
                double d = x_sorted[j] - y_sorted[j];
                jack[i] += d*d;
            }
        }
    }

    return Rcpp::List::create(Rcpp::Named("transp_cost") = cost, 
                              Rcpp::Named("jack_data")   = jack);
}