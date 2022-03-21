// Solve assignment problems using the network simplex algorithm

#include <Rcpp.h>
#include "network_simplex_simple.h"

using namespace Rcpp;
using namespace std;
using namespace lemon;


//' Solve the assignment problem using the network simplex from Bonneel et al, 2011
//'
//' Input: n-by-n cost matrix C. 
//'
//' @export
//[[Rcpp::export]]
double SolveAssignmentNetworkflow(const Rcpp::NumericMatrix &C)
{
    int64_t n = C.rows();

    // Set up the fully connected graph for the problem
    FullBipartiteDigraph di(n, n);
    NetworkSimplexSimple<FullBipartiteDigraph, double, double, long long> net(di, true, 2 * n, n * n);

    // Set the arc costs
    int64_t idarc = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
          double d = C(i, j);
          net.setCost(di.arcFromId(idarc), d);
          idarc++;
        }
    }

    // Set the node weights
    vector<double> weights1(n), weights2(n);
    for (int i = 0; i < n; i++)
    {
        weights1[di.nodeFromId(i)] = 1.;
        weights2[di.nodeFromId(i)] = -1.;
    }

    // Initialize the algorithm, then run it
    net.supplyMap(&weights1[0], n, &weights2[0], n);
    net.run();

    // Divide by the sample size
    return net.totalCost() / (double)n;
}
