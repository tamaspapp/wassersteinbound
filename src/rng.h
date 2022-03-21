/* This header links to the random number generator, which is intialized exactly once for all samplers when the package is loaded.*/

#ifndef RNG_H
#define RNG_H

#include <Rcpp.h>
#include <xoshiro.h>
#include <dqrng_distribution.h>
// [[Rcpp::depends(dqrng, BH, sitmo)]]

using boost::random::normal_distribution; // This Boost function uses the Ziggurat method, much faster than R's inversion.
using dqrng::uniform_distribution;
using dqrng::xoshiro256plus;

namespace RNG
{
    // Declare rng
    extern xoshiro256plus rng;

    // Declare distributions
    extern uniform_distribution runif;
    extern normal_distribution<double> rnorm; 
    
} // namesepace RNG

void SetSeed_cpp (int seed, int stream);

#endif /* RNG_H */