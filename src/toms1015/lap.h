#pragma once

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

#ifndef lapAlloc
#define lapAlloc lap::alloc
#endif

#ifndef lapFree
#define lapFree lap::free
#endif

#include <tuple>

namespace lap
{
	template <class SC> void solve(int &dim, int &dim2, SC* cost_matrix, int *rowsol, SC* u, SC* v, SC &totalcost, bool &use_epsilon);

	// Memory management
	template <typename T> void alloc(T * &ptr, unsigned long long width, const char *file, const int line);
	template <typename T> void free(T *&ptr);
}

#include "lap_solver.h"
