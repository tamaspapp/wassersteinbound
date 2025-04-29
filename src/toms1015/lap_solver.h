#pragma once

#include <chrono>
#include <sstream>
#include <iostream>
#include <cstring>
#ifndef LAP_QUIET
#include <deque>
#include <mutex>
#endif
#include <math.h>

namespace lap
{
	template <typename T>
	void alloc(T * &ptr, unsigned long long width, const char *file, const int line)
	{
		ptr = (T*)malloc(sizeof(T) * (size_t) width); // this one is allowed
	}

	template <typename T>
	void free(T *&ptr)
	{
		if (ptr == (T *)NULL) return;
		::free(ptr); // this one is allowed
		ptr = (T *)NULL;
	}

	template <class SC, typename COST>
	void getMinMaxBest(int i, SC &min_cost_l, SC &max_cost_l, SC &picked_cost_l, int &j_min, COST &cost, int *taken, int count)
	{
		max_cost_l = min_cost_l = cost(0);
		if (taken[0] == 0)
		{
			j_min = 0;
			picked_cost_l = min_cost_l;
		}
		else
		{
			j_min = std::numeric_limits<int>::max();
			picked_cost_l = std::numeric_limits<SC>::max();
		}
		for (int j = 1; j < count; j++)
		{
			SC cost_l = cost(j);
			min_cost_l = std::min(min_cost_l, cost_l);
			if (i == j) max_cost_l = cost_l;
			if ((cost_l < picked_cost_l) && (taken[j] == 0))
			{
				j_min = j;
				picked_cost_l = cost_l;
			}
		}
	}

	template <class SC, typename COST>
	void getMinSecondBest(SC &min_cost_l, SC &second_cost_l, SC &picked_cost_l, int &j_min, COST &cost, int *taken, int count)
	{
		min_cost_l = std::min(cost(0), cost(1));
		second_cost_l = std::max(cost(0), cost(1));
		if ((taken[0] == 0) && (taken[1] == 0))
		{
			picked_cost_l = min_cost_l;
			if (cost(0) == min_cost_l)
			{
				j_min = 0;
			}
			else
			{
				j_min = 1;
			}
		}
		else if (taken[0] == 0)
		{
			j_min = 0;
			picked_cost_l = cost(0);
		}
		else if (taken[1] == 0)
		{
			j_min = 1;
			picked_cost_l = cost(1);
		}
		else
		{
			j_min = std::numeric_limits<int>::max();
			picked_cost_l = std::numeric_limits<SC>::max();
		}
		for (int j = 2; j < count; j++)
		{
			SC cost_l = cost(j);
			if (cost_l < min_cost_l)
			{
				second_cost_l = min_cost_l;
				min_cost_l = cost_l;
			}
			else second_cost_l = std::min(second_cost_l, cost_l);
			if ((cost_l < picked_cost_l) && (taken[j] == 0))
			{
				j_min = j;
				picked_cost_l = cost_l;
			}
		}
	}

	template <class SC, typename COST>
	void updateEstimatedV(SC* v, SC *min_v, COST &cost, bool first, bool second, SC min_cost_l, SC max_cost_l, int count)
	{
		if (first)
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				min_v[j] = tmp;
			}
		}
		else if (second)
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				if (tmp < min_v[j])
				{
					v[j] = min_v[j];
					min_v[j] = tmp;
				}
				else v[j] = tmp;
			}
		}
		else
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				if (tmp < min_v[j])
				{
					v[j] = min_v[j];
					min_v[j] = tmp;
				}
				else v[j] = std::min(v[j], tmp);
			}
		}
	}

	template <class SC>
	void normalizeV(SC *v, int count, int *colsol)
	{
		SC max_v = std::numeric_limits<SC>::lowest();
		for (int j = 0; j < count; j++) if (colsol[j] >= 0) max_v = std::max(max_v, v[j]);
		for (int j = 0; j < count; j++) v[j] = std::min(SC(0), v[j] - max_v);
	}

	template <class SC>
	void normalizeV(SC *v, int count)
	{
		SC max_v = v[0];
		for (int j = 1; j < count; j++) max_v = std::max(max_v, v[j]);
		for (int j = 0; j < count; j++) v[j] = v[j] - max_v;
	}

	template <class SC, typename COST>
	void getMinimalCost(int &j_min, SC &min_cost, SC &min_cost_real, COST &cost, SC *mod_v, int count)
	{
		j_min = std::numeric_limits<int>::max();
		min_cost = std::numeric_limits<SC>::max();
		min_cost_real = std::numeric_limits<SC>::max();
		for (int j = 0; j < count; j++)
		{
			SC cost_l = cost(j);
			if (mod_v[j] < SC(0))
			{
				if (cost_l < min_cost)
				{
					min_cost = cost_l;
					j_min = j;
				}
			}
			min_cost_real = std::min(min_cost_real, cost_l);
		}
	}

	template<typename SC>
	void getUpperLower(SC& upper, SC& lower, double greedy_gap, double initial_gap, int dim, int dim2)
	{
		greedy_gap = std::min(greedy_gap, initial_gap / 4.0);
		if (greedy_gap < 1.0e-6 * initial_gap) upper = SC(0);
		else upper = (SC)((double)dim * greedy_gap * sqrt(greedy_gap / initial_gap) / ((double)dim2 * (double)dim2)); // Careful: can divide by zero!
		lower = (SC)(initial_gap / (16.0 * (double)dim2 * (double)dim2));
		if (upper < lower) upper = lower = SC(0);
	}

	template <class SC>
	std::pair<SC, SC> estimateEpsilon(int &dim, int &dim2, SC *cost_matrix, SC *v, int *perm) // With fix: solved numerical instability for cost matrices with small entries.
	{
		SC *mod_v;
		int *picked;
		SC *v2;

		lapAlloc(mod_v, dim2, __FILE__, __LINE__);
		lapAlloc(v2, dim2, __FILE__, __LINE__);
		lapAlloc(picked, dim2, __FILE__, __LINE__);

		double lower_bound = 0.0;
		double greedy_bound = 0.0;
		double upper_bound = 0.0;

		memset(picked, 0, sizeof(int) * dim2);

		for (int i = 0; i < dim2; i++)
		{
			SC min_cost_l, max_cost_l, picked_cost_l;
			int j_min;
			if (i < dim)
			{
					//const auto *tt = iterator.getRow(i);
					//auto cost = [&tt](int j) -> SC { return (SC)tt[j]; };
				auto cost = [&](int j) -> SC { return (SC)cost_matrix[i * dim + j]; }; // row-major order
				getMinMaxBest(i, min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked, dim2);
				picked[j_min] = 1;
				updateEstimatedV(v, mod_v, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, dim2);
				lower_bound += min_cost_l;
				upper_bound += max_cost_l;
				greedy_bound += picked_cost_l;
			}
			else
			{
				auto cost = [](int j) -> SC { return SC(0); };
				getMinMaxBest(i, min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked, dim2);
				picked[j_min] = 1;
				updateEstimatedV(v, mod_v, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, dim2);
				lower_bound += min_cost_l;
				greedy_bound += picked_cost_l;
			}
		}
		// make sure all j are < 0
		normalizeV(v, dim2);

		greedy_bound = std::min(greedy_bound, upper_bound);

		double initial_gap = upper_bound - lower_bound;
		double greedy_gap = greedy_bound - lower_bound;
		double initial_greedy_gap = greedy_gap;

		memset(picked, 0, sizeof(int) * dim2);

		lower_bound = 0.0;
		upper_bound = 0.0;

		// reverse order
		for (int i = dim2 - 1; i >= 0; --i)
		{
			SC min_cost_l, second_cost_l, picked_cost_l;
			int j_min;
			if (i < dim)
			{
					//const auto *tt = iterator.getRow(i);
					//auto cost = [&tt, &v](int j) -> SC { return (SC)tt[j] - v[j]; };
				auto cost = [&](int j) -> SC { return (SC)cost_matrix[i * dim + j] - v[j]; };
				getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked, dim2);
			}
			else
			{
				auto cost = [&v](int j) -> SC { return -v[j]; };
				getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked, dim2);
			}
			perm[i] = i;
			picked[j_min] = 1;
			mod_v[i] = second_cost_l - min_cost_l;
			// need to use the same v values in total
			lower_bound += min_cost_l + v[j_min];
			upper_bound += picked_cost_l + v[j_min];
		}

		upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

		greedy_gap = upper_bound - lower_bound;

		if (initial_gap < 4.0 * greedy_gap)
		{
			memcpy(v2, v, dim2 * sizeof(SC));
			// sort permutation by keys
			std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

			lower_bound = 0.0;
			upper_bound = 0.0;
			// greedy search
			std::fill(mod_v, mod_v + dim2, SC(-1));
			for (int i = 0; i < dim2; i++)
			{
				// greedy order
				int j_min;
				SC min_cost, min_cost_real;
				if (i < dim)
				{
						//const auto *tt = iterator.getRow(perm[i]);
						//auto cost = [&tt, &v](int j) -> SC { return (SC)tt[j] - v[j]; };
					auto cost = [&](int j) -> SC { return (SC)cost_matrix[perm[i] * dim + j] - v[j]; };
					getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v, dim2);
				}
				else
				{
					auto cost = [&v](int j) -> SC { return -v[j]; };
					getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v, dim2);
				}
				upper_bound += min_cost + v[j_min];
				// need to use the same v values in total
				lower_bound += min_cost_real + v[j_min];
				mod_v[j_min] = SC(0);
				picked[i] = j_min;
			}
			greedy_gap = upper_bound - lower_bound;

			// update v in reverse order
			for (int i = dim2 - 1; i >= 0; --i)
			{
				if (perm[i] < dim)
				{
						//const auto *tt = iterator.getRow(perm[i]);
						//SC min_cost = (SC)tt[picked[i]] - v[picked[i]];
					auto tt = [&](int j) -> SC { return (SC)cost_matrix[perm[i] * dim + j]; };
					SC min_cost = (SC)tt(picked[i]) - v[picked[i]];
					mod_v[picked[i]] = SC(-1);
					for (int j = 0; j < dim2; j++)
					{
						if (mod_v[j] >= SC(0))
						{
								//SC cost_l = (SC)tt[j] - v[j];
							SC cost_l = (SC)tt(j) - v[j];
							if (cost_l < min_cost) v[j] -= min_cost - cost_l;
						}
					}
				}
				else
				{
					SC min_cost = -v[picked[i]];
					mod_v[picked[i]] = SC(-1);
					for (int j = 0; j < dim2; j++)
					{
						if (mod_v[j] >= SC(0))
						{
							SC cost_l = -v[j];
							if (cost_l < min_cost) v[j] -= min_cost - cost_l;
						}
					}
				}
			}

			normalizeV(v, dim2);

			double old_upper_bound = upper_bound;
			double old_lower_bound = lower_bound;
			upper_bound = 0.0;
			lower_bound = 0.0;
			for (int i = 0; i < dim2; i++)
			{
				SC min_cost, min_cost_real;
				if (perm[i] < dim)
				{
						//const auto *tt = iterator.getRow(perm[i]);
						//min_cost = (SC)tt[picked[i]];
					auto tt = [&](int j) -> SC { return (SC)cost_matrix[perm[i] * dim + j]; };
					min_cost = (SC)tt(picked[i]);
					min_cost_real = std::numeric_limits<SC>::max();
					for (int j = 0; j < dim2; j++)
					{
							//SC cost_l = (SC)tt[j] - v[j];
						SC cost_l = (SC)tt(j) - v[j];
						min_cost_real = std::min(min_cost_real, cost_l);
					}
				}
				else
				{
					min_cost = SC(0);
					min_cost_real = std::numeric_limits<SC>::max();
					for (int j = 0; j < dim2; j++) min_cost_real = std::min(min_cost_real, -v[j]);
				}
				// need to use all picked v for the lower bound as well
				upper_bound += min_cost;
				lower_bound += min_cost_real + v[picked[i]];
			}
			upper_bound = std::min(upper_bound, old_upper_bound);
			lower_bound = std::max(lower_bound, old_lower_bound);
			greedy_gap = upper_bound - lower_bound;

			double ratio2 = greedy_gap / initial_greedy_gap; // Careful: can divide by zero!
			if (ratio2 > 1.0e-09)
			{
				for (int i = 0; i < dim2; i++)
				{
					v[i] = (SC)((double)v2[i] * ratio2 + (double)v[i] * (1.0 - ratio2));
				}
			}
		}

		SC upper, lower;
		getUpperLower(upper, lower, greedy_gap, initial_gap, dim, dim2);

		// Fix numerical instability for cost matrices with small entries
		if(isnan(upper)) upper = lower; // (upper == nan) when we divide by zero above

		lapFree(mod_v);
		lapFree(picked);
		lapFree(v2);

		return std::pair<SC, SC>((SC)upper, (SC)lower);
	}

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

	__forceinline void dijkstraCheck(int& endofpath, bool& unassignedfound, int jmin, int* colsol, char* colactive)
	{
		colactive[jmin] = 0;
		if (colsol[jmin] < 0)
		{
			endofpath = jmin;
			unassignedfound = true;
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(char* colactive, int start, int end, SC min, SC* v, SC* d)
	{
		for (int j1 = start; j1 < end; j1++)
		{
			if (colactive[j1] == 0)
			{
				SC dlt = min - d[j1];
				v[j1] -= dlt;
			}
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(char* colactive, int start, int end, SC min, SC* v, SC* d, SC eps, SC& total_d, SC& total_eps)
	{
		for (int j1 = start; j1 < end; j1++)
		{
			if (colactive[j1] == 0)
			{
				SC dlt = min - d[j1];
				total_d += dlt;
				total_eps += eps;
				v[j1] -= dlt + eps;
			}
		}
	}

	__forceinline void resetRowColumnAssignment(int &endofpath, int f, int *pred, int *rowsol, int *colsol)
	{
		int i;
		do
		{
			i = pred[endofpath];
			colsol[endofpath] = i;
			std::swap(endofpath, rowsol[i]);
		} while (i != f);
	}

	template <class SC, class TC>
	void getNextEpsilon(TC &epsilon, TC &epsilon_lower, SC &total_d, SC &total_eps, bool &first, bool &second, int &dim2)
	{
		if (epsilon > TC(0))
		{
			if (!first)
			{
				if ((TC(0.5) == TC(0)) && (epsilon == epsilon_lower)) epsilon = TC(0);
				else
				{
					if ((!second) && (total_d > total_eps))
					{
						epsilon = TC(0);
					}
					else
					{
						epsilon = std::min(epsilon / TC(4), (TC)(total_eps / SC(8 * (size_t)dim2)));
					}

					if (epsilon < epsilon_lower)
					{
						if (TC(0.5) == TC(0)) epsilon = epsilon_lower;
						else epsilon = TC(0);
					}
				}
			}
		}
	}

	template <class SC>
	void solve(int &dim, int &dim2, SC* cost_matrix, 
			   int *rowsol, SC* u, SC* v, // Output
	           const bool &use_epsilon)

		// input:
		// dim         - number of rows
		// dim2        - number of columns; we assume that dim2 >= dim
		// cost_matrix - cost matrix as a C array in ROW-MAJOR order, i.e. C(i,j) = cost_matrix[i * dim + j].
		// use_epsilon - whether to use the epsilon-scaling heuristic

		// output:
		// rowsol     - the assignment, i.e. the column assigned to each row
		// u          - dual variables, row potentials    (i.e. "row prices", i.e. "row reduction numbers")
		// v          - dual variables, column potentials (i.e. "column prices", i.e. "column reduction numbers")

	{
		int *pred;
		int endofpath;
		char *colactive;
		SC *d;
		int *colsol;      // colsol     - row assigned to column in solution
		SC epsilon_upper;
		SC epsilon_lower;
		int *perm;

		lapAlloc(colactive, dim2, __FILE__, __LINE__);
		lapAlloc(d, dim2, __FILE__, __LINE__);
		lapAlloc(pred, dim2, __FILE__, __LINE__);
		lapAlloc(colsol, dim2, __FILE__, __LINE__);
		lapAlloc(perm, dim2, __FILE__, __LINE__);

		SC epsilon;

		if (use_epsilon)
		{
			std::pair<SC, SC> eps = estimateEpsilon(dim, dim2, cost_matrix, v, perm);
			epsilon_upper = eps.first;
			epsilon_lower = eps.second;
		}
		else
		{
			memset(v, 0, dim2 * sizeof(SC)); // "Fill with 0"
			epsilon_upper = SC(0);
			epsilon_lower = SC(0);
		}
		epsilon = epsilon_upper;

		bool first = true;
		bool second = false;
		bool reverse = true;

		if ((!use_epsilon) || (epsilon > SC(0)))
		{
			for (int i = 0; i < dim2; i++) perm[i] = i;
			reverse = false;
		}

		SC total_d = SC(0);
		SC total_eps = SC(0);
		while (epsilon >= SC(0))
		{
			getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);

			total_d = SC(0);
			total_eps = SC(0);

			// this is to ensure termination of the while statement
			if (epsilon == SC(0)) epsilon = SC(-1.0);
			memset(rowsol, -1, dim2 * sizeof(int));
			memset(colsol, -1, dim2 * sizeof(int));
			int jmin, jmin_n;
			SC min, min_n;
			bool unassignedfound;
			int dim_limit = dim2;

			// AUGMENT SOLUTION for each free row.
			for (int fc = 0; fc < dim_limit; fc++)
			{
				int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];

				unassignedfound = false;

				// Dijkstra search
				min = std::numeric_limits<SC>::max();
				jmin = dim2;
				if (f < dim)
				{
						// auto tt = iterator.getRow(f);
					auto tt = [&](int j) -> SC { return (SC)cost_matrix[f * dim + j]; };
					for (int j = 0; j < dim2; j++)
					{
						colactive[j] = 1;
						pred[j] = f;
							//SC h = d[j] = tt[j] - v[j];
						SC h = d[j] = tt(j) - v[j];
						if (h <= min)
						{
							if (h < min)
							{
								// better
								jmin = j;
								min = h;
							}
							else //if (h == min)
							{
								// same, do only update if old was used and new is free
								if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
							}
						}
					}
				}
				else
				{
					for (int j = 0; j < dim2; j++)
					{
						colactive[j] = 1;
						pred[j] = f;
						SC h = d[j] = -v[j];
						if (colsol[j] < dim)
						{
							if (h <= min)
							{
								if (h < min)
								{
									// better
									jmin = j;
									min = h;
								}
								else //if (h == min)
								{
									// same, do only update if old was used and new is free
									if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
								}
							}
						}
					}
				}

				dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive);
				// marked skipped columns that were cheaper
				if (f >= dim)
				{
					for (int j = 0; j < dim2; j++)
					{
						// ignore any columns assigned to virtual rows
						if ((colsol[j] >= dim) && (d[j] <= min))
						{
							colactive[j] = 0;
						}
					}
				}

				while (!unassignedfound)
				{
					// update 'distances' between freerow and all unscanned columns, via next scanned column.
					int i = colsol[jmin];

					jmin_n = dim2;
					min_n = std::numeric_limits<SC>::max();
					if (i < dim)
					{
							//auto tt = iterator.getRow(i);
							//SC tt_jmin = (SC)tt[jmin];
						auto tt = [&](int j) -> SC { return (SC)cost_matrix[i * dim + j]; };
						SC tt_jmin = (SC)tt(jmin);
						SC v_jmin = v[jmin];
						for (int j = 0; j < dim2; j++)
						{
							if (colactive[j] != 0)
							{
									//SC v2 = (tt[j] - tt_jmin) - (v[j] - v_jmin) + min;
								SC v2 = (tt(j) - tt_jmin) - (v[j] - v_jmin) + min;
								SC h = d[j];
								if (v2 < h)
								{
									pred[j] = i;
									d[j] = v2;
									h = v2;
								}
								if (h <= min_n)
								{
									if (h < min_n)
									{
										// better
										jmin_n = j;
										min_n = h;
									}
									else //if (h == min_n)
									{
										// same, do only update if old was used and new is free
										if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
									}
								}
							}
						}
					}
					else
					{
						SC v_jmin = v[jmin];
						for (int j = 0; j < dim2; j++)
						{
							if (colactive[j] != 0)
							{
								SC v2 = -(v[j] - v_jmin) + min;
								SC h = d[j];
								if (v2 < h)
								{
									pred[j] = i;
									d[j] = v2;
									h = v2;
								}
								if (h <= min_n)
								{
									if (colsol[j] < dim)
									{
										if (h < min_n)
										{
											// better
											jmin_n = j;
											min_n = h;
										}
										else //if (h == min_n)
										{
											// same, do only update if old was used and new is free
											if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
										}
									}
								}
							}
						}
					}

					min = std::max(min, min_n);
					jmin = jmin_n;
					dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive);
					// marked skipped columns that were cheaper
					if (i >= dim)
					{
						for (int j = 0; j < dim2; j++)
						{
							// ignore any columns assigned to virtual rows
							if ((colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min))
							{
								colactive[j] = 0;
							}
						}
					}
				}

				// update column prices. can increase or decrease
				if (epsilon > SC(0))
				{
					updateColumnPrices(colactive, 0, dim2, min, v, d, epsilon, total_d, total_eps);
				}
				else
				{
					updateColumnPrices(colactive, 0, dim2, min, v, d);
				}

				// reset row and column assignments along the alternating path.
				resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
			}

#ifdef LAP_MINIMIZE_V
			if (epsilon > SC(0))
			{
#if 0
				if (dim_limit < dim2) normalizeV(v, dim2, colsol);
				else normalizeV(v, dim2);
#else
				if (dim_limit < dim2) for (int i = 0; i < dim2; i++) if (colsol[i] < 0) v[i] -= SC(2) * epsilon;
				normalizeV(v, dim2);
#endif
			}
#endif

			second = first;
			first = false;
			reverse = !reverse;
		}

		// Get row prices
		for (int i = 0; i < dim; i++)
		{
			u[i] = cost_matrix[i * dim +  rowsol[i]] - v[rowsol[i]];
		}

		// free reserved memory.
		lapFree(pred);
		lapFree(colactive);
		lapFree(d);
		lapFree(colsol);
		lapFree(perm);
	}
}
