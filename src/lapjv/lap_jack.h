#include <vector>
#include <limits>

template <typename cost>
void lap_jack(const int &dim, cost *assigncost_, int *rowsol_, cost *u, cost *v_, const cost &totalcost, const cost &small, cost* jack_costs)

// input:
// dim         - problem size
// assigncost_ - cost matrix, C array in row-major order
// rowsol_     - column assigned to row in solution, i.e. the assignment
// u           - dual variables, row reduction numbers
// v_          - dual variables, column reduction numbers
// totalcost   - total assignment cost
// small       - cost value small enough to guarantee assignment

// output:
// jack_costs - jackknife transportation costs, entry "i" leaves sample "i" out in both the row and the column
{
    bool unassignedfound;
    int i, f, k, freerow, *pred;
    int j, j1, endofpath, last, low, up, *collist;
    cost min, h, v2, *d;

    collist = new int[dim];  // list of columns to be scanned in various ways.
    d = new cost[dim];     // 'cost-distance' in augmenting path calculation.
    pred = new int[dim];     // row-predecessor of column in augmenting/alternating path.

	std::vector<int> colsol_(dim);
	for(int idx = 0; idx < dim; idx++)
	{
		colsol_[rowsol_[idx]] = idx;
	}

	for(int removed = 0; removed < dim; removed++)
	{
		if(removed == rowsol_[removed])
		{
			jack_costs[removed] = totalcost - assigncost_[removed * dim + removed];
			continue;
		}
		// Removing a sample -- same as making the right edge have a very small cost, then solving for new assignment
		auto assigncost = [&] (int idx1, int idx2) -> cost {return (idx1 == removed && idx2 == removed) ? small : assigncost_[idx1 * dim + idx2];};

// Proceed with the "dynamic" version of the LAPJV algorithm
 		int rm_col = removed;
		int rm_row = colsol_[removed];
		// Unmatch the edge associated to the removed column
		std::vector<int> rowsol(rowsol_, rowsol_ + dim);
		rowsol[rm_row] = -1;
		auto colsol = [&] (int idx) -> int {return (idx == rm_col) ? -1 : colsol_[idx];};

		// Restore complementary slackness by changing "column price" v
		cost v_removed = std::numeric_limits<cost>::max();
		for (int i = 0; i < dim; i++)
		{
			v_removed = std::min(assigncost(i, removed) - u[i], v_removed);
		}
		auto v = [&] (int idx) -> cost {return (idx == removed) ? v_removed : v_[idx]; };

		// AUGMENT SOLUTION for each free row. (Only one free row.)
		{
			// start row of augmenting path.
			freerow = rm_row;
			// Dijkstra shortest path algorithm.
			// runs until unassigned column added to shortest path tree.
			for (j = 0; j < dim; j++)
			{
				d[j] = assigncost(freerow, j)  - v(j);
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
						if (colsol(collist[k]) < 0)
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
					i = colsol(j1);
					h = assigncost(i, j1) - v(j1) - min;

					for (k = up; k < dim; k++)
					{
						j = collist[k];
						v2 = assigncost(i, j) - v(j) - h;
						if (v2 < d[j])
						{
							pred[j] = i;
							if (v2 == min) // new column found at same minimum value
								if (colsol(j) < 0)
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

			// reset row and column assignments along the alternating path.
			do
			{
				i = pred[endofpath];
				j1 = endofpath;
				endofpath = rowsol[i];
				rowsol[i] = j1;
			} while (i != freerow);
		}

		jack_costs[removed] = 0;
		for (int idx = 0; idx < dim; idx++)
        {
        	if (idx != removed)
            {
                jack_costs[removed] += assigncost_[idx * dim + rowsol[idx]];
            }
        }
	}

    // Free memory
    delete[] pred;
    delete[] collist;
    delete[] d;
} 

