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
#include "rng_pcg.h"

namespace RNG
{
    pcg32 rng(12345);
    uniform_distribution runif(0., 1.); // Boost function
    normal_distribution rnorm(0., 1.);  // Ziggurat method, much faster than R's inversion.
    // Poisson: we'll define rpois inline for Scalable MH, as the rate needs to change.

    Eigen::VectorXd GaussianVector(const int &d)
    {
        Eigen::VectorXd result(d);
        for (int i = 0; i < d; i++)
        {
            result[i] = RNG::rnorm(RNG::rng);
        }
        return result;
    }

    // Uniform integer from 0 to (N-1). (From: https://www.pcg-random.org/posts/bounded-rands.html)
    uint32_t bounded_rand(pcg32 &rng, uint32_t range)
    {
        uint32_t x = rng();
        uint64_t m = uint64_t(x) * uint64_t(range);
        uint32_t l = uint32_t(m);
        if (l < range)
        {
            uint32_t t = -range;
            if (t >= range)
            {
                t -= range;
                if (t >= range)
                    t %= range;
            }
            while (l < t)
            {
                x = rng();
                m = uint64_t(x) * uint64_t(range);
                l = uint32_t(m);
            }
        }
        return m >> 32;
    }

    // Knuth's Algorithm 3.4.2S in "Seminumeric Algorithms", from: https://stackoverflow.com/a/311716/15485
    std::vector<int> SampleWithoutReplacement(int populationSize, int sampleSize)
    {
        // Use Knuth's variable names
        int &n = sampleSize;
        int &N = populationSize;

        int t = 0; // total input records dealt with
        int m = 0; // number of items selected so far
        double u;

        std::vector<int> samples(n);
        while (m < n)
        {
            u = RNG::runif(RNG::rng); // call a uniform(0,1) random number generator

            if ((N - t) * u >= n - m)
            {
                t++;
            }
            else
            {
                samples[m] = t;
                t++;
                m++;
            }
        }
        return samples;
    }

} // namespace RNG


//' Re-Seed RNG from R
//'
//' @export
// [[Rcpp::export(rng = false)]]
void SetSeed_pcg32(const int &seed, const int &stream = 0)
{
    pcg32 rng_refresh(seed, stream);
    RNG::rng = rng_refresh;
}
