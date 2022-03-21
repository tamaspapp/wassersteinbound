#include <Rcpp.h>
#include "network_simplex_simple.h"

#define EPS 1E-12 // Tolerance for checking equality between doubles

using std::vector;
using std::min_element;
using std::max_element;

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;
using Rcpp::Named;

using lemon::FullBipartiteDigraph;
using lemon::NetworkSimplexSimple;

// Given a previously computed optimal solution for the assignment problem (including duals),
// re-solve to obtain the optimal assignment with the edge "removed"--"removed" removed (or fixed).
//
// As per Mills-Tettey et al, The Dynamic Hungarian Algorithm for the Assignment Problem with Changing Costs, 2007
// https://www.cs.cmu.edu/~gertrude/dyn_assign_techreport.pdf
//
    // input:
    // dim        - problem size
    // assigncost - cost matrix
    // removed    - index of row to be removed
    // small      - a cost small enough to guarantee assignment between any row and column. Used to remove edge removed-->removed.

    // output:
    // rowsol     - column assigned to row in solution. 
    //
    // colsol     - row assigned to column in solution
    // u          - dual variables, row reduction numbers
    // v          - dual variables, column reduction numbers

void lap_jack(const int &dim, 
              vector<double> &assigncost, 
              vector<int> &rowsol, vector<int> &colsol, 
              vector<double> &u,   vector<double> &v, 
              const int &removed, 
              const double &small)
{
    bool unassignedfound;
    int i, numfree, f, k, freerow, *pred, *free;
    int j, j1, endofpath, last, low, up, *collist;
    double min, h, v2, *d;

    free = new int[dim];     // list of unassigned rows.
    collist = new int[dim];  // list of columns to be scanned in various ways.
    d = new double[dim];     // 'cost-distance' in augmenting path calculation.
    pred = new int[dim];     // row-predecessor of column in augmenting/alternating path.

 // Proceed with the "dynamic" version of the LAPJV algorithm

    // Remove the column
    int rm_col = removed;
    int rm_row = colsol[removed];
    free[0] = rm_row; // Remove row from matching

    // Unmatch the edge associated to the removed column (+row)
    rowsol[rm_row] = -1;
    colsol[rm_col] = -1;

    // Restore feasbility by changing "column price" v
    double find_min = assigncost[0 * dim + removed] - u[0];
    double temp;
    for (int i = 1; i < dim; i++)
    {   
        temp = assigncost[i * dim + removed] - u[i];
        if (temp < find_min)
        {
            find_min = temp;
        }
    }
    v[removed] = find_min;

    numfree = 1;

    // AUGMENT SOLUTION for each free row.
    for (f = 0; f < numfree; f++)
    {
        freerow = free[f]; // start row of augmenting path.

        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for (j = 0; j < dim; j++)
        {
            d[j] = assigncost[freerow * dim + j] - v[j];
            pred[j] = freerow;
            collist[j] = j; // init column list.
        }

        low = 0; // columns in 0..low-1 are ready, now none.
        up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                 // columns in up..dim-1 are to be considered later to find new minimum,
                 // at this stage the list simply contains all columns
        unassignedfound = false;
        do
        {
            if (up == low) // no more columns to be scanned for current minimum.
            {
                last = low - 1;

                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                min = d[collist[up++]];
                for (k = up; k < dim; k++)
                {
                    j = collist[k];
                    h = d[j];
                    if (h <= min)
                    {
                        if (h < min) // new minimum.
                        {
                            up = low; // restart list at index low.
                            min = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }

                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for (k = low; k < up; k++)
                    if (colsol[collist[k]] < 0)
                    {
                        endofpath = collist[k];
                        unassignedfound = true;
                        break;
                    }
            }

            if (!unassignedfound)
            {
                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                j1 = collist[low];
                low++;
                i = colsol[j1];
                h = assigncost[i * dim + j1] - v[j1] - min;

                for (k = up; k < dim; k++)
                {
                    j = collist[k];
                    v2 = assigncost[i * dim + j] - v[j] - h;
                    if (v2 < d[j])
                    {
                        pred[j] = i;
                        if (v2 + EPS > min && min + EPS > v2) // new column found at same minimum value
                            if (colsol[j] < 0)
                            {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassignedfound = true;
                                break;
                            }
                            // else add to list to be scanned right away.
                            else
                            {
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        d[j] = v2;
                    }
                }
            }
        } while (!unassignedfound);

        // update column prices.
        for (k = 0; k <= last; k++)
        {
            j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        // reset row and column assignments along the alternating path.
        do
        {
            i = pred[endofpath];
            colsol[endofpath] = i;
            j1 = endofpath;
            endofpath = rowsol[i];
            rowsol[i] = j1;
        } while (i != freerow);
    }

    // Free memory
    delete[] pred;
    delete[] free;
    delete[] collist;
    delete[] d;
}


//' The Flapjack algorithm
//'
//' Input: n-by-n cost matrix C
//'
//' @export
//[[Rcpp::export(rng = false)]]
Rcpp::List Flapjack(const Rcpp::NumericMatrix &C)
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

    // Get the optimal transport cost
    double lapcost = net.totalCost();

 // Jackknife

    // Get dual variables
    vector<double> u(n), v(n);
    for (int i = 0; i < n; i++)
    {
        u[i] = (-1) * net.potential(i);// +  0.001 * i;
        v[i] = net.potential(i + n);// +  0.001 * i;
    }

    /* Unnecessary step: 
    // Pre-empt any issues with Dijkstra: force all of the edges in the shortest path search to be non-negative, while preserving optimality.
    double v_max = *std::max_element(v.begin(), v.end());
    for (int i = 0; i < n; i++)
    {
        u[i] = u[i] + v_max;
        v[i] = v[i] - v_max;
    }*/
   
    // Get row and column assignments
    vector<int> rowsol(n), colsol(n);
    
    double flow;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            flow = net.flow(di.arcFromId(i * n + j));
            if (flow > 0.5) // In case the flow isn't exactly 1 on assigned edges
            {
                rowsol[i] = j;
                colsol[j] = i;
            }
        }
    }

    // Set up the cost matrix as a vector
    vector<double> assigncost(n * n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            assigncost[i * n + j] = C(i, j);// + 0.001 * (i + j);
        }
    }

    // Find a value that's definitely small enough to guarantee assignment
    double max = *max_element(assigncost.begin(), assigncost.end());
    double min = *min_element(assigncost.begin(), assigncost.end());
    double small = 2 * min - max - 1;
    
    // Compute the (un-normalized) Jackknife transport costs 
    vector<double> jack(n, 0);

    vector<int> rowsol_new(n), colsol_new(n);
    vector<double> u_new(n), v_new(n);

    for (int i = 0; i < n; i++) 
    {
        // If the removed edge is in the optimal matching, don't have to recompute anything.
        // So bypass the computation entirely.
        if (rowsol[i] == i)
        {
            jack[i] = lapcost - C(i, i);
        }
        else
        {
            // Make the edge cost small enough so that it is always in the optimal matching
            assigncost[i * n + i] = small; 

            // Re-start with original potentials and assignment
            rowsol_new = rowsol;
            colsol_new = colsol;
            u_new = u; 
            v_new = v;

            lap_jack(n, assigncost, rowsol_new, colsol_new, u_new, v_new, i, small);
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                {
                    jack[i] += C(j, rowsol_new[j]);
                }
            }

            assigncost[i * n + i] =  C(i, i); // Re-assign the original costs
        }
    }

    // Divide by the sample size to obtain the transportation costs
    return Rcpp::List::create(Rcpp::Named("transp_cost") = lapcost / (double) n, 
                              Rcpp::Named("jack_data")   = Rcpp::NumericVector(jack.begin(), jack.end()) / (double) (n - 1.));
}
