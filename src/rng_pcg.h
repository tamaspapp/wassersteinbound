/* This header links to the random number generator, which is intialized exactly once for all samplers when the package is loaded. */

#ifndef RNG_PCG_H
#define RNG_PCG_H

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]
#include <cmath>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>

using uniform_distribution = boost::random::uniform_real_distribution<double>;
using normal_distribution = boost::random::normal_distribution<double>;
using poisson_distribution = boost::random::poisson_distribution<int>;

#include "rng/pcg_random.hpp"
#include "rng/pcg_extras.hpp"

namespace RNG
{
    extern pcg32 rng;
    extern uniform_distribution runif;
    extern normal_distribution rnorm;

    extern Eigen::VectorXd GaussianVector(const int &d);

    extern uint32_t bounded_rand(pcg32 &rng, uint32_t range);

    extern std::vector<int> SampleWithoutReplacement(int populationSize, int sampleSize);

} // namespace RNG

void SetSeed_cpp(const int &seed, const int &stream);

#endif /* RNG_PCG_H */
