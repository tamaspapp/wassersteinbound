// Solve assignment problems

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp/Benchmark/Timer.h>

#include "cost_eval.h"  // Cost matrix evaluation
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> RowMajorMat;

#include "lapjv/lap_jack.h" // Jonker-Volgenant for jackknife
#include "toms1015/lap.h"   // TOMS1015 linear assignment problem solver
#define LAP_MINIMIZE_V
#include "networksimplex/network_simplex_simple.h" // Bonneel et al. (2011) network simplex solver

//' Solve the assignment problem using TOMS1015, Guthe and Thuerck (2021)
//'
//' Input: n-by-d empirical measures x and y.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List assignment_squared_euclidean_toms1015_cpp(const Eigen::Map<Eigen::MatrixXd> &x,
                                                     const Eigen::Map<Eigen::MatrixXd> &y,
                                                     const bool &estimate_epsilon = true)
{
    int N = x.rows();
    Rcpp::Timer timer; // Rcpp timer, default in nanoseconds
    timer.step("start");

    // Calculate cost matrix
    RowMajorMat C = squared_euclidean_cost_matrix_rowmajor(x, y);
    timer.step("cost_matrix");

    // Solve assignment problem
    Rcpp::IntegerVector assignment(N, -1);
    Rcpp::NumericVector row_prices(N), column_prices(N);
    double total_cost = 0;
    Rcpp::NumericVector cost_fractions(N);
    lap::solve<double>(N, N, C.data(), assignment.begin(), row_prices.begin(), column_prices.begin(), estimate_epsilon);
    for(int i = 0; i < N; i++)
    {
        cost_fractions(i) = C(i, assignment(i));
        total_cost += cost_fractions(i);
    }
    timer.step("assignment");

    Rcpp::NumericVector timing(timer);
    return Rcpp::List::create(Rcpp::Named("cost") = total_cost, // Divide by N to get the squared Wasserstein distance
                              Rcpp::Named("cost_fractions") = cost_fractions,
                              Rcpp::Named("row_potentials") = row_prices,
                              Rcpp::Named("column_potentials") = column_prices,
                              Rcpp::Named("assignment") = assignment + 1, // Index from 1 for R output
                              Rcpp::Named("timing_nanoseconds") = timing);
}

//' Solve the assignment problem using the network simplex from Bonneel et al. (2011)
//'
//' Input: n-by-d empirical measures x and y.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List assignment_squared_euclidean_networkflow_cpp(const Eigen::Map<Eigen::MatrixXd> &x,
                                                        const Eigen::Map<Eigen::MatrixXd> &y,
                                                        const bool &compute_assignment = false) // Whether to compute the assignment. This is a non-trivial O(n^2) cost for the network flow algorithm: we need to scan all arcs.
{
    int64_t N = x.rows();
    Rcpp::Timer timer; // Rcpp timer, default in nanoseconds
    timer.step("start");

    // Evaluate cost matrix
    RowMajorMat C = squared_euclidean_cost_matrix_rowmajor(x, y);
    timer.step("cost_matrix");

    // Solve assigment problem
    lemon::FullBipartiteDigraph di(N, N); // Set up the fully connected graph for the problem
    lemon::NetworkSimplexSimple<lemon::FullBipartiteDigraph, double, double, long long> net(di, true, 2 * N, N * N);
    int64_t idarc = 0; // Set the arc costs.
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
          double d = C(i, j);
          net.setCost(di.arcFromId(idarc), d);
          idarc++;
        }
    }
    net.supplyMapAll(+1., N, -1., N); // Set the node weights, +1 for source and -1 for sink
    net.run();  // Solve
    timer.step("assignment");

    // Output cost, OT potentials, and assignment
    double cost = net.totalCost();
    Rcpp::NumericVector potentials_source(N);
    Rcpp::NumericVector potentials_sink(N);
    for (int i = 0; i < N; i++)
    {
        potentials_source(i) = (-1.) * net.potential(i);
        potentials_sink(i) = net.potential(i + N);
    }
    Rcpp::IntegerVector assignment(N, -1);
    Rcpp::NumericVector cost_fractions(N);
    if (compute_assignment) // O(n^2) cost to compute the assignment!
    {
        for (int i = 0; i < N; i++) // Row
        {
            for (int j = 0; j < N; j++) // Column
            {
                if (net.flow(di.arcFromId(i * N + j)) > 0.5) // Constant is chosen arbitrarily; the flow in the n-by-n case is either 0 (unassigned) or 1 (assigned).
                {
                    assignment(i) = j;
                    cost_fractions(i) = C(i, assignment(i));
                    break;
                }
            }
        }
        timer.step("output");
    }

    Rcpp::NumericVector timing(timer);
    return Rcpp::List::create(Rcpp::Named("cost") = cost,
                              Rcpp::Named("cost_fractions") = cost_fractions,
                              Rcpp::Named("row_potentials") = potentials_source,
                              Rcpp::Named("column_potentials") = potentials_sink,
                              Rcpp::Named("assignment") = assignment + 1, // R indexes from 1.
                              Rcpp::Named("timing_nanoseconds") = timing);
}


//' Solve the assignment problem using TOMS1015, Guthe and Thuerck (2021), then obtain the jackknife assignment costs using the "vanilla" Jonker-Volgenant algorithm
//'
//' Input: n-by-d empirical measures x and y.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List assignment_squared_euclidean_jackknife_cpp(const Eigen::Map<Eigen::MatrixXd> &x,
                                                      const Eigen::Map<Eigen::MatrixXd> &y,
                                                      const bool &estimate_epsilon = true)
{
    int N = x.rows();
    Rcpp::Timer timer; // Rcpp timer, default in nanoseconds
    timer.step("start");

    // Calculate cost matrix
    RowMajorMat C = squared_euclidean_cost_matrix_rowmajor(x, y);
    timer.step("cost_matrix");

    // Solve assignment problem
    Rcpp::IntegerVector assignment(N, -1);
    Rcpp::NumericVector row_prices(N), column_prices(N);
    double total_cost = 0;
    Rcpp::NumericVector cost_fractions(N);
    lap::solve<double>(N, N, C.data(), assignment.begin(), row_prices.begin(), column_prices.begin(), estimate_epsilon);
    for(int i = 0; i < N; i++)
    {
        cost_fractions(i) = C(i, assignment(i));
        total_cost += cost_fractions(i);
    }
    timer.step("assignment");

    Rcpp::NumericVector jack_costs(N);
    double small = 2 * C.minCoeff() - C.maxCoeff() - 0.1; // small enough to guarantee assignment
    // double small = -0.1 * std::numeric_limits<double>::max();
    lap_jack<double>(N, C.data(), assignment.begin(), row_prices.begin(), column_prices.begin(), total_cost, small, jack_costs.begin());
    timer.step("jackknife");

    Rcpp::NumericVector timing(timer);
    return Rcpp::List::create(Rcpp::Named("cost") = total_cost, // Divide by N to get the squared Wasserstein distance
                              Rcpp::Named("cost_fractions") = cost_fractions,
                              Rcpp::Named("row_potentials") = row_prices,
                              Rcpp::Named("column_potentials") = column_prices,
                              Rcpp::Named("assignment") = assignment + 1, // Index from 1 for R output
                              Rcpp::Named("timing_nanoseconds") = timing,
                              Rcpp::Named("jack_costs") = jack_costs);
}

/*********** One-dimensional case ********/

// Compute the one-dimensional squared-Euclidean transportation cost, together with leave-one-out transportation costs used to compute a jackknife error estimate
//
// input:
// x_sorted    = Order statistics of 1d sample x. (Increasing order is assumed, decreasing order should also work.)
// inv_order_x = The inverse of the permutation which had sorted sample x. I.e. the location of the sample X_0, then X_1, and so on.
//               Indexing must start from 0!
//
// This is NOT optimized.
//
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::List assignment_squared_euclidean_jackknife_1d_cpp(const Rcpp::NumericVector &x_sorted, 
                                                         const Rcpp::NumericVector &y_sorted, 
                                                         const Rcpp::IntegerVector &inv_order_x,  // (!) In R indexing, from 1.
                                                         const Rcpp::IntegerVector &inv_order_y)
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
            min = inv_order_y[i] - 1; // Go from R indexing to C indexing
            max = inv_order_x[i] - 1;
            
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
            min = inv_order_x[i] - 1; // Go from R indexing to C indexing
            max = inv_order_y[i] - 1;

            // Skip "min" in x, then skip "max" in y
            for (int j = 0; j < min - 1; j++)
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

    return Rcpp::List::create(Rcpp::Named("cost") = cost, // Divide by N to get the squared Wasserstein distance
                              Rcpp::Named("jack_costs") = jack);
}

