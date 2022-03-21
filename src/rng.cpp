/* The random number generator used by all samplers. */

/*
Set up random number generation such that:
  (a) Results are reprodicible;
  (b) In R, random number generation carries on from where it left over previously;
  (c) By default, the same stream is used for all random number generation;
  (d) Multiple independent streams can be used for parallel random number generation.
  
This is acheived by defining the RNG in its own namespace, which is compiled together with all of the random number generating functions.
*/

#include <Rcpp.h>
#include <xoshiro.h>
#include <dqrng_distribution.h>
// [[Rcpp::depends(dqrng, BH, sitmo)]]

#include "rng.h"

namespace RNG
{
    xoshiro256plus rng(12345);
    uniform_distribution runif(0., 1.);
    normal_distribution<double> rnorm(0., 1.); // This Boost function uses the Ziggurat method, much faster than R's inversion.

} // namespace RNG


// Re-Seed RNG from R
// 
// seed = RNG seed
// stream = RNG stream, useful for parallel random number generation
//
//' @export
//[[Rcpp::export(rng = false)]]
void SetSeed_cpp(const int &seed, const int &stream = 0)
{
    xoshiro256plus rng_refresh(seed);

    if (stream != 0)
    {
        rng_refresh.long_jump(stream);
    }

    RNG::rng = rng_refresh;
}
