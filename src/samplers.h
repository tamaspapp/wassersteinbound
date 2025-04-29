#include <cmath>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "targets.h"
#include "rng_pcg.h"

using Eigen::Lower;
using Eigen::Upper;

using Eigen::MatrixBase;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::accumulate;
using std::vector;

/************* Helper functions -- RNG **********/

namespace RNG
{
    class Alias
    {
    private:
        pcg32 &rng_;
        std::vector<double> q_;
        std::vector<int> j_;

    public:
        Alias(pcg32 &rng, const std::vector<double> &probs)
            : rng_(rng)
        {
            int K = probs.size();
            double sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0);

            for (int i = 0; i < K; i++)
            {
                q_.push_back(K * probs[i] / sum_probs);
                j_.push_back(i);
            }

            std::vector<double> g;
            std::vector<double> s;
            for (int i = 0; i < K; i++)
            {
                if (q_[i] > 1.)
                {
                    g.push_back(i);
                }
                else
                {
                    s.push_back(i);
                }
            }

            while (g.size() > 0 && s.size() > 0)
            {
                int k = g.back();
                int l = s.back();
                s.pop_back();

                j_[l] = k;
                q_[k] += q_[l] - 1.0;

                if (q_[k] < 1.0)
                {
                    g.pop_back();
                    s.push_back(k);
                }
            }
        }

        int Sample()
        {
            std::vector<int> rv;
            int X = RNG::bounded_rand(rng_, q_.size());
            double V = RNG::runif(rng_);
            if (V < q_[X])
            {
                return X;
            }
            else
            {
                return j_[X];
            }
        }

        std::vector<int> Sample(const int &N)
        {
            std::vector<int> rv;
            for (int i = 0; i < N; i++)
            {
                rv.push_back(Sample());
            }
            return rv;
        }
    };
}

/************* Helper functions -- linear algebra **********/

// Cholesky factorization
inline Eigen::MatrixXd chol_l(const Eigen::Ref<const Eigen::MatrixXd> &Sigma)
{
    if (Sigma.cols() == 1)
        return Sigma.cwiseSqrt();
    else
        return Sigma.llt().matrixL();
}
inline Eigen::MatrixXd chol_u(const Eigen::Ref<const Eigen::MatrixXd> &Sigma)
{
    if (Sigma.cols() == 1)
        return Sigma.cwiseSqrt();
    else
        return Sigma.llt().matrixU();
}

// Multiply by dense matrix
inline Eigen::VectorXd precondition(const Eigen::Ref<const Eigen::MatrixXd> &Sigma, 
                                    const Eigen::Ref<const Eigen::VectorXd> &x)
{
    if (Sigma.cols() == 1)
        return Sigma.cwiseProduct(x);
    else
        return Sigma * x;
}

// Multiply by triangular matrices
inline Eigen::VectorXd precondition_lowertri(const Eigen::Ref<const Eigen::MatrixXd> &L, 
                                             const Eigen::Ref<const Eigen::VectorXd> &x)
{
    if (L.cols() == 1)
        return L.cwiseProduct(x);
    else
        return L.triangularView<Lower>() * x;
}
inline Eigen::VectorXd precondition_uppertri(const Eigen::Ref<const Eigen::MatrixXd> &U, 
                                             const Eigen::Ref<const Eigen::VectorXd> &x)
{
    if (U.cols() == 1)
        return U.cwiseProduct(x);
    else
        return U.triangularView<Upper>() * x;
}
inline Eigen::VectorXd precondition_l_top(const Eigen::Ref<const Eigen::MatrixXd> &L, 
                                          const Eigen::Ref<const Eigen::VectorXd> &x)
{
    if (L.cols() == 1)
        return L.cwiseProduct(x);
    else
        return L.triangularView<Lower>().transpose() * x;
}

// Multiply by inverses of triangular matrices
inline Eigen::VectorXd precondition_l_inverse(const Eigen::Ref<const Eigen::MatrixXd> &L, 
                                              const Eigen::Ref<const Eigen::VectorXd> &x)
{
    if (L.cols() == 1)
        return x.cwiseProduct(L.cwiseInverse());
    else
        return L.triangularView<Lower>().solve(x); // Uses Gaussian elimination, as fast as multiplying by the actual inverse
}
inline Eigen::VectorXd precondition_u_inverse(const Eigen::Ref<const Eigen::MatrixXd> &U, 
                                              const Eigen::Ref<const Eigen::VectorXd> &x)
{
    if (U.cols() == 1)
        return x.cwiseProduct(U.cwiseInverse());
    else
        return U.triangularView<Upper>().solve(x); // Uses Gaussian elimination, as fast as multiplying by the actual inverse
}

/************* Helper functions -- MCMC **********/

// Gaussian proposal, using lower triangular Cholesky factor
inline Eigen::VectorXd gaussian_proposal(const Eigen::Ref<const Eigen::VectorXd> &x,
                                         const Eigen::Ref<const Eigen::MatrixXd> &L,
                                         const Eigen::Ref<const Eigen::VectorXd> &z)
{
    return x + precondition_lowertri(L, z);
}
inline Eigen::VectorXd mala_proposal_mean(const Target &target,
                                          const Eigen::Ref<const Eigen::VectorXd> &x,
                                          const Eigen::Ref<const Eigen::MatrixXd> &Sigma)
{
    return x + 0.5 * precondition(Sigma, target.GradientLogDensity(x));
}

// Proposal log density difference log q(x | xp) - log q(xp | x) for MALA.
//      - See Proposition 1 in Titsias (2023, NeurIPS).
//      - This form saves two matmuls, at the expense of storing two O(d) temporaries x_mean and xp_mean.
inline double mala_proposal_logdensity_difference(const Eigen::Ref<const Eigen::VectorXd> &x, const Eigen::Ref<const Eigen::VectorXd> &x_mean, const Eigen::Ref<const Eigen::VectorXd> &x_grad,
                                                  const Eigen::Ref<const Eigen::VectorXd> &xp, const Eigen::Ref<const Eigen::VectorXd> &xp_mean, const Eigen::Ref<const Eigen::VectorXd> &xp_grad)
{
    return 0.5 * (x - 0.5 * (xp + xp_mean)).dot(xp_grad) - 0.5 * (xp - 0.5 * (x + x_mean)).dot(x_grad);
}

// Minus the kinetic energy for Hamiltonian dynamics with inverse-mass matrix Sigma
//      - i.e. the density of a mean-zero Gaussian with PRECISION matrix Sigma
//      - parametrized in terms of the upper Cholesky factor of the inverse-mass matrix
//
inline double minus_kinetic_energy(const Eigen::Ref<const Eigen::VectorXd> &v,
                                   const Eigen::Ref<const Eigen::MatrixXd> &Sigma_chol_u) // Upper Cholesky factor of the inverse-mass matrix
{
    return (-0.5) * precondition_uppertri(Sigma_chol_u, v).squaredNorm();
}

/************* Helper functions -- tall data MCMC **********/

inline VectorXd subsampled_gradpotential_noreplace(const Eigen::Ref<const Eigen::VectorXd> &x,
                                                   const TargetTallData &target,
                                                   const int &batch_size)
{
    int d = x.size();
    int n = target.Count();
    double rescale = (double)n / (double)batch_size;

    vector<int> idx = RNG::SampleWithoutReplacement(n, batch_size);

    VectorXd x_grad_potential = VectorXd::Zero(d);
    for (const int &i : idx)
        x_grad_potential += target.GradientPotential(i, x);
    x_grad_potential *= rescale;

    return x_grad_potential;
}

inline VectorXd subsampled_gradpotential_noreplace_cv(const Eigen::Ref<const Eigen::VectorXd> &x,
                                                      const TargetTallData &target,
                                                      const int &batch_size,
                                                      const Eigen::Ref<const Eigen::VectorXd> &gradient_at_mode, // Gradient of the log-POTENTIAL at the mode
                                                      const std::vector<Eigen::VectorXd> &gradients_at_mode)     // Gradients of the log-POTENTIAL at the mode, per each sample
{
    int d = x.size();
    int n = target.Count();
    double rescale = (double)n / (double)batch_size;

    vector<int> idx = RNG::SampleWithoutReplacement(n, batch_size);

    VectorXd x_grad_potential = VectorXd::Zero(d);
    for (const int &i : idx)
        x_grad_potential += target.GradientPotential(i, x) - gradients_at_mode[i];
    x_grad_potential *= rescale;
    x_grad_potential += gradient_at_mode;

    return x_grad_potential;
}

/************* Single-chain MCMC **********/

template <typename m1>
void rwm(const Target &target,
         const Eigen::Ref<const Eigen::VectorXd> &x0,
         const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // Cholesky factor of proposal covariance. Can also input a vector for diagonal covariances.
         const int &iter,
         const int &thin,
         Eigen::MatrixBase<m1> &xs,
         double &acceptance_rate)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma);

    Eigen::VectorXd x = x0;
    double logpi_x = target.LogDensity(x);
    xs.row(0) = x;

    int accepted = 0;
    for (int it = 1; it <= iter; it++)
    {
        Eigen::VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));

        Eigen::VectorXd xp = x + Lz;
        double logpi_xp = target.LogDensity(xp);

        // Metropolis correction
        double logHR_x = logpi_xp - logpi_x;
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            ++accepted;
            x = xp;
            logpi_x = logpi_xp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
    }
    acceptance_rate = (double)accepted / (double)iter;
}

template <typename m1>
void mala(const Target &target,
          const Eigen::Ref<const Eigen::VectorXd> &x0,
          const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // Poposal covariance. Can also be a vector for diagonal covariances.
          const int &iter,
          const int &thin,
          Eigen::MatrixBase<m1> &xs,
          double &acceptance_rate)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma);

    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target.GradientLogDensity(x);
    Eigen::VectorXd x_mean = x + 0.5 * precondition(Sigma, x_grad);
    double logpi_x = target.LogDensity(x);
    xs.row(0) = x;

    int accepted = 0;
    for (int it = 1; it <= iter; it++)
    {
        Eigen::VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));

        Eigen::VectorXd xp = x_mean + Lz;
        Eigen::VectorXd xp_grad = target.GradientLogDensity(xp);
        Eigen::VectorXd xp_mean = xp + 0.5 * precondition(Sigma, xp_grad);
        double logpi_xp = target.LogDensity(xp);

        // Metropolis correction
        double logHR_x = logpi_xp - logpi_x + mala_proposal_logdensity_difference(x, x_mean, x_grad, xp, xp_mean, xp_grad);
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            ++accepted;
            x = xp;
            x_grad = xp_grad;
            x_mean = xp_mean;
            logpi_x = logpi_xp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
    }
    acceptance_rate = (double)accepted / (double)iter;
}

template <typename m1>
void ula(const Target &target,
         const Eigen::Ref<const Eigen::VectorXd> &x0,
         const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // Poposal covariance. Can also be a vector for diagonal covariances.
         const int &iter,
         const int &thin,
         Eigen::MatrixBase<m1> &xs)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma);

    Eigen::VectorXd x = x0;
    xs.row(0) = x;

    for (int it = 1; it <= iter; it++)
    {
        Eigen::VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));
        
        x = x + 0.5 * precondition(Sigma, target.GradientLogDensity(x)) + Lz;

        if (it % thin == 0)
            xs.row(it / thin) = x;
    }
}

template <typename m1>
void obab(const Target &target,
          const Eigen::Ref<const Eigen::VectorXd> &x0,
          const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // INVERSE mass matrix
          const double &eta,                              // OU process "persistency" multipler for timestep delta
          const double &delta,                            // Timestep
          const int &iter,
          const int &thin,
          Eigen::MatrixBase<m1> &xs)
{
    int d = x0.size();
    Eigen::MatrixXd Sigma_chol_u = chol_u(Sigma);

    Eigen::VectorXd v = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d));
    Eigen::VectorXd x = x0;
    xs.row(0) = x;

    for (int it = 1; it <= iter; it++)
    {
        // O full-step
        Eigen::VectorXd z = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d));
        v = eta * v + sqrt(1 - eta * eta) * z;

        // B half-step
        v = v + 0.5 * delta * target.GradientLogDensity(x);

        // A full-step
        x = x + delta * precondition(Sigma, v);

        // B half-step
        v = v + 0.5 * delta * target.GradientLogDensity(x);

        if (it % thin == 0)
            xs.row(it / thin) = x;
    }
}

template <typename m1>
void horowitz(const Target &target,
              const Eigen::Ref<const Eigen::VectorXd> &x0,
              const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
              double &eta,
              const double &delta,
              const int &iter,
              const int &thin,
              Eigen::MatrixBase<m1> &xs,
              std::vector<bool> &acceptances)
{
    int d = x0.size();
    Eigen::MatrixXd Sigma_chol_u = chol_u(Sigma);

    Eigen::VectorXd v = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)), z(d);
    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target.GradientLogDensity(x);
    double logpi_x = target.LogDensity(x);
    xs.row(0) = x;

    for (int it = 1; it <= iter; it++)
    {
        // O full-step
        z = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)); // Sample refreshment
        v = eta * v + sqrt(1 - eta * eta) * z;

        // Start trajectory
        Eigen::VectorXd vp = v;
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xp_grad = x_grad;

        // B half-step
        double minus_KE_init = minus_kinetic_energy(vp, Sigma_chol_u);
        vp += 0.5 * delta * xp_grad;

        // A full-step
        xp += delta * precondition(Sigma, vp);
        xp_grad = target.GradientLogDensity(xp);

        // B half-step
        vp += 0.5 * delta * xp_grad;
        double minus_KE_final = minus_kinetic_energy(vp, Sigma_chol_u);

        // Metropolis correction
        double logpi_xp = target.LogDensity(xp);
        double logHR_x = logpi_xp + minus_KE_final - logpi_x - minus_KE_init;
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            v = vp;
            x = xp;
            x_grad = xp_grad;
            logpi_x = logpi_xp;
            acceptances[it - 1] = true;
        }
        else // Turn around
        {
            v = (-v).eval();
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
    }
}

/************* Adaptive MCMC **********/

// For Fisher-MALA, see Proposition 4 in Titsias (2023, NeurIPS)
inline void matrix_sqrt_update(Eigen::Ref<Eigen::MatrixXd> R,              // "Root", will be changed in-place
                               const Eigen::Ref<const Eigen::VectorXd> &s) // "Signal"
{
    Eigen::VectorXd phi = R.transpose() * s;
    Eigen::VectorXd Rphi = R * phi;

    double phi_sq = phi.squaredNorm();                // ||phi||^2
    double divisor = (1 + phi_sq + sqrt(1 + phi_sq)); // Divisor of the low-rank update is: 1 + ||phi||^2 + sqrt(1 + ||phi||^2)

    R -= (Rphi / divisor) * phi.transpose();
}

Eigen::MatrixXd init_matrix_sqrt_update(const Eigen::Ref<const Eigen::VectorXd> &s, // "Signal"
                                        const double &lambda)                       // Damping parameter
{
    int d = s.size();
    double s_sq = s.squaredNorm();                                                    // ||s||^2
    double divisor = sqrt(lambda) * (lambda + s_sq + sqrt(lambda * (lambda + s_sq))); // Divisor of the low-rank update is: sqrt(lambda) * (lambda + ||s||^2 + sqrt(lambda * (lambda + ||s||^2)))

    return Eigen::MatrixXd::Identity(d, d) / sqrt(lambda) - (s / divisor) * s.transpose();
}

template <typename m1>
void fisher_mala(const Target &target,
                 const Eigen::Ref<const Eigen::VectorXd> &x0,
                 double &sigma,
                 Eigen::Ref<Eigen::MatrixXd> R, // Eigen::MatrixXd R = Eigen::MatrixXd::Identity(d, d);
                 const double &learning_rate,   // For global scale parameter "sigma"
                 const double &acc_rate_target, // i.e. 0.574
                 const double &damping,
                 const int &iter,
                 const int &thin,
                 Eigen::MatrixBase<m1> &xs,
                 std::vector<bool> &acceptances)
{
    int d = x0.size();
    double sigmaR = sigma;
    double sigmaR_sq = sigmaR * sigmaR; // Multiplier in front of R * R.transpose()
    double sigma_sq = sigma * sigma;    // Global scale for normalized preconditioner; dictates the acceptance rate

    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target.GradientLogDensity(x);
    Eigen::VectorXd x_mean(d);
    double logpi_x = target.LogDensity(x);
    xs.row(0) = x;

    int accepted = 0;
    for (int it = 1; it <= iter; it++)
    {
        // Need to re-evaluate all of "x_mean" at the start of every iteration, since we adapt the covariance even if we reject the proposal.
        x_mean = x + (0.5 * sigmaR_sq) * (R * (R.transpose() * x_grad)); // With more care, we could save on these two matvecprods.

        // Update proposal
        Eigen::VectorXd z = RNG::GaussianVector(d);
        Eigen::VectorXd xp = x_mean + sqrt(sigmaR_sq) * (R * z); // This is correct, since we refreshed x_mean at the start of the iteration.
        Eigen::VectorXd xp_grad = target.GradientLogDensity(xp);
        Eigen::VectorXd xp_mean = xp + (0.5 * sigmaR_sq) * (R * (R.transpose() * xp_grad));
        double logpi_xp = target.LogDensity(xp);

        // Metropolis correction
        double logHR_x = logpi_xp - logpi_x + mala_proposal_logdensity_difference(x, x_mean, x_grad, xp, xp_mean, xp_grad);
        double acc_prob = std::min(1., exp(logHR_x));
        if (RNG::runif(RNG::rng) < acc_prob)
        {
            acceptances[it - 1] = true;
            x = xp;
            x_grad = xp_grad;
            logpi_x = logpi_xp;
        }
        if (it % thin == 0)
            xs.row(it / thin) = x;

        // Adapt matrix square-root
        Eigen::VectorXd s_delta = sqrt(acc_prob) * (xp_grad - x_grad);
        if (it == 1)
            R = init_matrix_sqrt_update(s_delta, damping);
        else
            matrix_sqrt_update(R, s_delta);

        // Adapt scale factor
        sigma_sq += sigma_sq * learning_rate * (acc_prob - acc_rate_target) * std::min(1., 100. / pow(it, 2/3)); // Effectively, start diminishing the adaptation after 1000 iterations.
        sigmaR_sq = sigma_sq / (R.squaredNorm() / (double)d); // Use that (R * R.transpose()).trace() = R.array().square().sum() = R.squaredNorm()
    }

    sigma = sqrt(sigma_sq);
    sigmaR = sqrt(sigmaR_sq);
}

/************* Tall data MCMC **********/

// SGLD, sampling the indices without replacement
template <typename m1>
void sgld(const TargetTallData &target,
          const int &batch_size,
          const Eigen::Ref<const Eigen::VectorXd> &theta0,
          const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // Volatility of each increment
          const int &iter,
          const int &thin,
          Eigen::MatrixBase<m1> &samples)
{
    int d = theta0.size();
    int n = target.Count();
    MatrixXd Sigma_chol_l = chol_l(Sigma); // Cholesky factorization, for proposal generation

    VectorXd current_theta = theta0;
    samples.row(0) = current_theta;
    for (int it = 1; it <= iter; it++)
    {
        VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));
        VectorXd grad_potential = subsampled_gradpotential_noreplace(current_theta, target, batch_size);
        current_theta = current_theta - 0.5 * precondition(Sigma, grad_potential) + Lz;

        if (it % thin == 0)
            samples.row(it / thin) = current_theta;
    }
}

// SGLD with control variates, sampling the indices without replacement
template <typename m1>
void sgldcv(const TargetTallData &target,
            const Eigen::Ref<const Eigen::VectorXd> &mode,
            const Eigen::Ref<const Eigen::VectorXd> &gradient_at_mode,
            const int &batch_size,
            const Eigen::Ref<const Eigen::VectorXd> &theta0,
            const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // Volatility of each increment
            const int &iter,
            const int &thin,
            Eigen::MatrixBase<m1> &samples)
{
    int d = theta0.size();
    int n = target.Count();
    MatrixXd Sigma_chol_l = chol_l(Sigma); // Cholesky factorization, for proposal generation

    vector<VectorXd> gradients_at_mode(n);
    for (int i = 0; i < n; i++)
        gradients_at_mode[i] = target.GradientPotential(i, mode);

    VectorXd current_theta = theta0;
    samples.row(0) = current_theta;
    for (int it = 1; it <= iter; it++)
    {
        VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));
        VectorXd grad_potential = subsampled_gradpotential_noreplace_cv(current_theta, target, batch_size, gradient_at_mode, gradients_at_mode);
        current_theta = current_theta - 0.5 * precondition(Sigma, grad_potential) + Lz;

        if (it % thin == 0)
            samples.row(it / thin) = current_theta;
    }
}

/************* Coupled MCMC, same kernel **********/

// Two-scale GCRN coupling, with reflection-maximal coupling when: thresh > || L.inverse() * (x - y) ||^2.
template <typename v1, typename v2, typename m1, typename m2, typename m3>
void rwm_twoscaleGCRN(const Target &target,
                      const Eigen::MatrixBase<v1> &x0,
                      const Eigen::MatrixBase<v2> &y0,
                      const Eigen::MatrixBase<m1> &Sigma, // Cholesky factor of proposal covariance. Can also input a vector for diagonal covariances.
                      const int &iter,
                      const int &thin,
                      const double &thresh,
                      Eigen::MatrixBase<m2> &xs,
                      Eigen::MatrixBase<m3> &ys,
                      double &acceptance_rate_x,
                      double &acceptance_rate_y,
                      int &tau)
{
    int d = x0.size();
    MatrixXd L = chol_l(Sigma);

    // X-chain
    Eigen::VectorXd x = x0, xp(d);
    double logpi_x = target.LogDensity(x), logpi_xp;
    bool update_gx = true;
    xs.row(0) = x;
    int accepted_x = 0;

    // Y-chain
    Eigen::VectorXd y = y0, yp(d);
    double logpi_y = target.LogDensity(y), logpi_yp;
    bool update_gy = true;
    ys.row(0) = y;
    int accepted_y = 0;

    // Will need these for the GCRN coupling.
    Eigen::VectorXd e_gx(d), e_gy(d);

    int iter_done = iter;
    for (int it = 1; it <= iter; it++)
    {
        if ((x-y).squaredNorm() < thresh) // Reflection-maximal
        {
             Eigen::VectorXd z = precondition_l_inverse(L, x - y);

            Eigen::VectorXd zx = RNG::GaussianVector(d);
            xp = gaussian_proposal(x, L, zx);
            logpi_xp = target.LogDensity(xp);

            double log_probcpl = -0.5 * (zx + z).squaredNorm() + 0.5 * zx.squaredNorm();
            double log_ucpl = log(RNG::runif(RNG::rng));
            if (log_probcpl > 0 || log_ucpl < log_probcpl) // Coalesce proposals
            {
                yp = xp;
                logpi_yp = logpi_xp;
            }
            else // Reflect
            {
                z.normalize();
                Eigen::VectorXd zy = zx - 2 * zx.dot(z) * z;
                yp = gaussian_proposal(y, L, zy);
                logpi_yp = target.LogDensity(yp);
            }
        }
        else // GCRN
        {
            if (update_gx)
            {
                e_gx = precondition_l_top(L, target.GradientLogDensity(x));
                e_gx.normalize();
                update_gx = false;
            }
            if (update_gy)
            {
                e_gy = precondition_l_top(L, target.GradientLogDensity(y));
                e_gy.normalize();
                update_gy = false;
            }

            Eigen::VectorXd z = RNG::GaussianVector(d);
            double z1 = RNG::rnorm(RNG::rng);

            Eigen::VectorXd zx = z + (z1 - z.dot(e_gx)) * e_gx;
            Eigen::VectorXd zy = z + (z1 - z.dot(e_gy)) * e_gy;

            xp = gaussian_proposal(x, L, zx);
            yp = gaussian_proposal(y, L, zy);

            logpi_xp = target.LogDensity(xp);
            logpi_yp = target.LogDensity(yp);
        }

        double logHR_x = logpi_xp - logpi_x;
        double logHR_y = logpi_yp - logpi_y;

        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            ++accepted_x;
            x = xp;
            logpi_x = logpi_xp;
            update_gx = true;
        }
        if (logHR_y > 0 || log_u < logHR_y)
        {
            ++accepted_y;
            y = yp;
            logpi_y = logpi_yp;
            update_gy = true;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;

        // Break out of the loop if we're coupled. As set up, the states ~will~ be exactly equal when coalescence occurs.
        if ((x.array() == y.array()).all())
        {
            tau = it;
            iter_done = it;
            break;
        }
    }

    acceptance_rate_x = (double)accepted_x / (double)iter_done;
    acceptance_rate_y = (double)accepted_y / (double)iter_done;
}

// Two-scale CRN coupling, with reflection-maximal coupling when: thresh > || L.inverse() * (x_mean - y_mean) ||^2, where L is the Cholesky factor of the proposal covariance (= Sigma_chol_l = Omega_chol_u)
template <typename m1, typename m2>
void mala_twoscaleCRN(const Target &target,
                      const Eigen::Ref<const Eigen::VectorXd> &x0,
                      const Eigen::Ref<const Eigen::VectorXd> &y0,
                      const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // See the definition of MALA for what these matrices are.
                      const int &iter,
                      const int &thin,
                      const double &thresh,
                      Eigen::MatrixBase<m1> &xs,
                      Eigen::MatrixBase<m2> &ys,
                      double &acceptance_rate_x,
                      double &acceptance_rate_y,
                      int &tau)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma);

    // X-chain
    Eigen::VectorXd x = x0, xp(d);
    Eigen::VectorXd x_grad = target.GradientLogDensity(x), xp_grad(d);
    Eigen::VectorXd x_mean = x + 0.5 * precondition(Sigma, x_grad), xp_mean(d);
    double logpi_x = target.LogDensity(x), logpi_xp;
    xs.row(0) = x;
    int accepted_x = 0;

    // Y-chain
    Eigen::VectorXd y = y0, yp(d);
    Eigen::VectorXd y_grad = target.GradientLogDensity(y), yp_grad(d);
    Eigen::VectorXd y_mean = y + 0.5 * precondition(Sigma, y_grad), yp_mean(d);
    double logpi_y = target.LogDensity(y), logpi_yp;
    ys.row(0) = y;
    int accepted_y = 0;

    int iter_done = iter;
    for (int it = 1; it <= iter; it++)
    {
        if ((x-y).squaredNorm() < thresh) // Reflection-maximal
        {
            Eigen::VectorXd z = precondition_l_inverse(Sigma_chol_l, x_mean - y_mean);

            Eigen::VectorXd zx = RNG::GaussianVector(d);
            xp = gaussian_proposal(x_mean, Sigma_chol_l, zx);
            xp_grad = target.GradientLogDensity(xp);
            xp_mean = xp + 0.5 * precondition(Sigma, xp_grad);
            logpi_xp = target.LogDensity(xp);

            double log_probcpl = -0.5 * (zx + z).squaredNorm() + 0.5 * zx.squaredNorm();
            double log_ucpl = log(RNG::runif(RNG::rng));
            if (log_probcpl > 0 || log_ucpl < log_probcpl) // Coalesce proposals
            {
                yp = xp;
                yp_grad = xp_grad;
                yp_mean = xp_mean;
                logpi_yp = logpi_xp;
            }
            else // Reflect
            {
                z.normalize();
                Eigen::VectorXd zy = zx - 2 * zx.dot(z) * z;
                yp = gaussian_proposal(y_mean, Sigma_chol_l, zy);
                yp_grad = target.GradientLogDensity(yp);
                yp_mean = yp + 0.5 * precondition(Sigma, yp_grad);
                logpi_yp = target.LogDensity(yp);
            }
        }
        else // CRN
        {
            Eigen::VectorXd z = RNG::GaussianVector(d);

            xp = gaussian_proposal(x_mean, Sigma_chol_l, z);
            xp_grad = target.GradientLogDensity(xp);
            xp_mean = xp + 0.5 * precondition(Sigma, xp_grad);
            logpi_xp = target.LogDensity(xp);

            yp = gaussian_proposal(y_mean, Sigma_chol_l, z);
            yp_grad = target.GradientLogDensity(yp);
            yp_mean = yp + 0.5 * precondition(Sigma, yp_grad);
            logpi_yp = target.LogDensity(yp);
        }

        double logHR_x = logpi_xp - logpi_x + mala_proposal_logdensity_difference(x, x_mean, x_grad, xp, xp_mean, xp_grad);
        double logHR_y = logpi_yp - logpi_y + mala_proposal_logdensity_difference(y, y_mean, y_grad, yp, yp_mean, yp_grad);

        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            ++accepted_x;
            x = xp;
            x_mean = xp_mean;
            x_grad = xp_grad;
            logpi_x = logpi_xp;
        }
        if (logHR_y > 0 || log_u < logHR_y)
        {
            ++accepted_y;
            y = yp;
            y_mean = yp_mean;
            y_grad = yp_grad;
            logpi_y = logpi_yp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;

        // Break out of the loop if we're coupled. As set up, the states ~will~ be exactly equal when coalescence occurs.
        if ((x.array() == y.array()).all())
        {
            tau = it;
            iter_done = it;
            break;
        }
    }

    acceptance_rate_x = (double)accepted_x / (double)iter_done;
    acceptance_rate_y = (double)accepted_y / (double)iter_done;
}

// Two-scale CRN coupling, with reflection-maximal coupling when: thresh > || L.inverse() * (x_mean - y_mean) ||^2, where L is the Cholesky factor of the proposal covariance (= Sigma_chol_l)
template <typename m1, typename m2>
void ula_twoscaleCRN(const Target &target,
                     const Eigen::Ref<const Eigen::VectorXd> &x0,
                     const Eigen::Ref<const Eigen::VectorXd> &y0,
                     const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
                     const int &iter,
                     const int &thin,
                     const double &thresh,
                     Eigen::MatrixBase<m1> &xs,
                     Eigen::MatrixBase<m2> &ys,
                     int &tau)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma);

    // X-chain
    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_mean(d);
    xs.row(0) = x;

    // Y-chain
    Eigen::VectorXd y = y0;
    Eigen::VectorXd y_mean(d);
    ys.row(0) = y;

    for (int it = 1; it <= iter; it++)
    {
        x_mean = mala_proposal_mean(target, x, Sigma);
        y_mean = mala_proposal_mean(target, y, Sigma);

        if ((x-y).squaredNorm() < thresh) // Reflection-maximal
        {
            Eigen::VectorXd z = precondition_l_inverse(Sigma_chol_l, x_mean - y_mean);

            Eigen::VectorXd zx = RNG::GaussianVector(d);
            x = gaussian_proposal(x_mean, Sigma_chol_l, zx);

            double log_probcpl = -0.5 * (zx + z).squaredNorm() + 0.5 * zx.squaredNorm();
            double log_ucpl = log(RNG::runif(RNG::rng));
            if (log_probcpl > 0 || log_ucpl < log_probcpl) // Coalesce the chains
            {
                y = x;
            }
            else // Reflect
            {
                z.normalize();
                Eigen::VectorXd zy = zx - (2 * zx.dot(z)) * z;
                y = gaussian_proposal(y_mean, Sigma_chol_l, zy);
            }
        }
        else // CRN
        {
            VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));
            x = x_mean + Lz;
            y = y_mean + Lz;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;

        // Break out of the loop if we're coupled. As set up, the states ~will~ be exactly equal when coalescence occurs.
        if ((x.array() == y.array()).all()) // Eigen only exists if they are exactly equal, see https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#title41
        {
            tau = it;
            break;
        }
    }
}

template <typename m1, typename m2>
void obab_CRN(const Target &target,
              const Eigen::Ref<const Eigen::VectorXd> &x0,
              const Eigen::Ref<const Eigen::VectorXd> &y0,
              const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
              const double &eta,
              const double &delta,
              const int &iter,
              const int &thin,
              Eigen::MatrixBase<m1> &xs,
              Eigen::MatrixBase<m2> &ys)
{
    int d = y0.size();
    Eigen::MatrixXd Sigma_chol_u = chol_u(Sigma);

    // X
    Eigen::VectorXd vx = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)), z(d);
    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target.GradientLogDensity(x);
    xs.row(0) = x;

    // Y
    Eigen::VectorXd vy = vx;
    Eigen::VectorXd y = y0;
    Eigen::VectorXd y_grad = target.GradientLogDensity(y);
    ys.row(0) = y;

    for (int it = 1; it <= iter; it++)
    {
        // O full-step
        z = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)); // Sample refreshment
        vx = eta * vx + sqrt(1 - eta * eta) * z;
        vy = eta * vy + sqrt(1 - eta * eta) * z;

        // B half-step
        vx += 0.5 * delta * target.GradientLogDensity(x);
        vy += 0.5 * delta * target.GradientLogDensity(y);

        // A full-step
        x += delta * precondition(Sigma, vx);
        y += delta * precondition(Sigma, vy);

        // B half-step
        vx += 0.5 * delta * target.GradientLogDensity(x);
        vy += 0.5 * delta * target.GradientLogDensity(y);

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }
}

template <typename m1, typename m2>
void horowitz_CRN(const Target &target,
                  const Eigen::Ref<const Eigen::VectorXd> &x0,
                  const Eigen::Ref<const Eigen::VectorXd> &y0,
                  const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
                  const double &eta,
                  const double &delta,
                  const int &iter,
                  const int &thin,
                  Eigen::MatrixBase<m1> &xs,
                  Eigen::MatrixBase<m2> &ys,
                  std::vector<bool> &acceptances_x,
                  std::vector<bool> &acceptances_y)
{
    int d = y0.size();
    Eigen::MatrixXd Sigma_chol_u = chol_u(Sigma);

    // X
    Eigen::VectorXd vx = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)), z(d);
    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target.GradientLogDensity(x);
    double logpi_x = target.LogDensity(x);
    xs.row(0) = x;

    // Y
    Eigen::VectorXd vy = vx;
    Eigen::VectorXd y = y0;
    Eigen::VectorXd y_grad = target.GradientLogDensity(y);
    double logpi_y = target.LogDensity(y);
    ys.row(0) = y;

    for (int it = 1; it <= iter; it++)
    {
        // O full-step
        z = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)); // Sample refreshment
        vx = eta * vx + sqrt(1 - eta * eta) * z;
        vy = eta * vy + sqrt(1 - eta * eta) * z;

        // Start trajectory
        Eigen::VectorXd vpx = vx, vpy = vy;
        Eigen::VectorXd xp = x, yp = y;
        Eigen::VectorXd xp_grad = x_grad, yp_grad = y_grad;

        // B half-step
        double minus_KE_x_init = minus_kinetic_energy(vpx, Sigma_chol_u);
        double minus_KE_y_init = minus_kinetic_energy(vpy, Sigma_chol_u);
        vpx += 0.5 * delta * xp_grad;
        vpy += 0.5 * delta * yp_grad;

        // A full-step
        xp += delta * precondition(Sigma, vpx);
        yp += delta * precondition(Sigma, vpy);
        xp_grad = target.GradientLogDensity(xp);
        yp_grad = target.GradientLogDensity(yp);

        // B half-step
        vpx += 0.5 * delta * xp_grad;
        vpy += 0.5 * delta * yp_grad;
        double minus_KE_x_final = minus_kinetic_energy(vpx, Sigma_chol_u);
        double minus_KE_y_final = minus_kinetic_energy(vpy, Sigma_chol_u);

        // Metropolis correction
        double logpi_xp = target.LogDensity(xp);
        double logpi_yp = target.LogDensity(yp);
        double logHR_x = logpi_xp + minus_KE_x_final - logpi_x - minus_KE_x_init;
        double logHR_y = logpi_yp + minus_KE_y_final - logpi_y - minus_KE_y_init;

        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x) // Accept
        {
            acceptances_x[it - 1] = true;
            vx = vpx;
            x = xp;
            x_grad = xp_grad;
            logpi_x = logpi_xp;
        }
        else // Turn around
        {
            vx = (-vx).eval();
        }
        if (logHR_y > 0 || log_u < logHR_y) // Accept
        {
            acceptances_y[it - 1] = true;
            vy = vpy;
            y = yp;
            y_grad = yp_grad;
            logpi_y = logpi_yp;
        }
        else // Turn around
        {
            vy = (-vy).eval();
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }
}

/************* Coupled MCMC, different kernels **********/

template <typename m1, typename m2>
void mala_CRN_2targets(const Target &target_x,
                       const Target &target_y,
                       const Eigen::Ref<const Eigen::VectorXd> &x0,
                       const Eigen::Ref<const Eigen::VectorXd> &y0,
                       const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
                       const int &iter,
                       const int &thin,
                       Eigen::MatrixBase<m1> &xs,
                       Eigen::MatrixBase<m2> &ys,
                       double &acceptance_rate_x,
                       double &acceptance_rate_y)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma); // "L"

    // X
    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target_x.GradientLogDensity(x);
    Eigen::VectorXd x_mean = x + 0.5 * precondition(Sigma, x_grad);
    double logpi_x = target_x.LogDensity(x);
    xs.row(0) = x;
    int accepted_x = 0;

    // Y
    Eigen::VectorXd y = y0;
    Eigen::VectorXd y_grad = target_y.GradientLogDensity(y);
    Eigen::VectorXd y_mean = y + 0.5 * precondition(Sigma, y_grad);
    double logpi_y = target_y.LogDensity(y);
    ys.row(0) = y;
    int accepted_y = 0;

    for (int it = 1; it <= iter; it++)
    {
        Eigen::VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d)); // For CRN coupling

        Eigen::VectorXd xp = x_mean + Lz;
        Eigen::VectorXd xp_grad = target_x.GradientLogDensity(xp);
        Eigen::VectorXd xp_mean = xp + 0.5 * precondition(Sigma, xp_grad);
        double logpi_xp = target_x.LogDensity(xp);

        Eigen::VectorXd yp = y_mean + Lz;
        Eigen::VectorXd yp_grad = target_y.GradientLogDensity(yp);
        Eigen::VectorXd yp_mean = yp + 0.5 * precondition(Sigma, yp_grad);
        double logpi_yp = target_y.LogDensity(yp);

        // Metropolis correction
        double logHR_x = logpi_xp - logpi_x + mala_proposal_logdensity_difference(x, x_mean, x_grad, xp, xp_mean, xp_grad);
        double logHR_y = logpi_yp - logpi_y + mala_proposal_logdensity_difference(y, y_mean, y_grad, yp, yp_mean, yp_grad);
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_x > 0 || log_u < logHR_x)
        {
            ++accepted_x;
            x = xp;
            x_grad = xp_grad;
            x_mean = xp_mean;
            logpi_x = logpi_xp;
        }
        if (logHR_y > 0 || log_u < logHR_y)
        {
            ++accepted_y;
            y = yp;
            y_grad = yp_grad;
            y_mean = yp_mean;
            logpi_y = logpi_yp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }
    acceptance_rate_x = (double)accepted_x / (double)iter;
    acceptance_rate_y = (double)accepted_y / (double)iter;
}

template <typename m1, typename m2>
void ula_mala_CRN(const Target &target,
                  const Eigen::Ref<const Eigen::VectorXd> &x0,
                  const Eigen::Ref<const Eigen::VectorXd> &y0,
                  const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
                  const int &iter,
                  const int &thin,
                  Eigen::MatrixBase<m1> &xs,
                  Eigen::MatrixBase<m2> &ys,
                  double &acceptance_rate_y)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma); // "L"

    // ULA
    Eigen::VectorXd x = x0;
    xs.row(0) = x;

    // MALA
    Eigen::VectorXd y = y0;
    Eigen::VectorXd y_grad = target.GradientLogDensity(y);
    Eigen::VectorXd y_mean = y + 0.5 * precondition(Sigma, y_grad);
    double logpi_y = target.LogDensity(y);

    ys.row(0) = y;
    int accepted_y = 0;

    for (int it = 1; it <= iter; it++)
    {
        Eigen::VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d)); // For CRN coupling

        Eigen::VectorXd x_grad = target.GradientLogDensity(x);
        Eigen::VectorXd x_mean = x + 0.5 * precondition(Sigma, x_grad);
        x = x_mean + Lz;

        Eigen::VectorXd yp = y_mean + Lz;
        Eigen::VectorXd yp_grad = target.GradientLogDensity(yp);
        Eigen::VectorXd yp_mean = yp + 0.5 * precondition(Sigma, yp_grad);
        double logpi_yp = target.LogDensity(yp);

        double logHR_y = logpi_yp - logpi_y + mala_proposal_logdensity_difference(y, y_mean, y_grad, yp, yp_mean, yp_grad);
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_y > 0 || log_u < logHR_y)
        {
            ++accepted_y;
            y = yp;
            y_grad = yp_grad;
            y_mean = yp_mean;
            logpi_y = logpi_yp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }

    acceptance_rate_y = (double)accepted_y / (double)iter;
}

template <typename m1, typename m2>
void obab_horowitz_CRN(const Target &target,
                       const Eigen::Ref<const Eigen::VectorXd> &x0,    // OBAB
                       const Eigen::Ref<const Eigen::VectorXd> &y0,    // Horowitz (OBAB + Metropolis)
                       const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // Inverse-mass matrix
                       const double &eta,
                       const double &delta,
                       const int &iter,
                       const int &thin,
                       Eigen::MatrixBase<m1> &xs,
                       Eigen::MatrixBase<m2> &ys,
                       std::vector<bool> &acceptances_y)
{
    int d = y0.size();
    Eigen::MatrixXd Sigma_chol_u = chol_u(Sigma);

    // X: OBAB
    Eigen::VectorXd vx = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)), z(d);
    Eigen::VectorXd x = x0;
    Eigen::VectorXd x_grad = target.GradientLogDensity(x);
    xs.row(0) = x;

    // Y: Horowitz
    Eigen::VectorXd vy = vx;
    Eigen::VectorXd y = y0;
    Eigen::VectorXd y_grad = target.GradientLogDensity(y);
    double logpi_y = target.LogDensity(y);
    ys.row(0) = y;

    for (int it = 1; it <= iter; it++)
    {
        z = precondition_u_inverse(Sigma_chol_u, RNG::GaussianVector(d)); // Sample refreshment

        // X: OBAB
        vx = eta * vx + sqrt(1 - eta * eta) * z; // O full-step

        vx += 0.5 * delta * target.GradientLogDensity(x); // B half-step
        x += delta * precondition(Sigma, vx);             // A full-step
        vx += 0.5 * delta * target.GradientLogDensity(x); // B half-step

        // Y: Horowitz
        vy = eta * vy + sqrt(1 - eta * eta) * z; // O full-step

        // Start trajectory
        Eigen::VectorXd vpy = vy;
        Eigen::VectorXd yp = y;
        Eigen::VectorXd yp_grad = y_grad;

        // B half-step
        double minus_KE_y_init = minus_kinetic_energy(vpy, Sigma_chol_u);
        vpy += 0.5 * delta * yp_grad;

        // A full-step
        yp += delta * precondition(Sigma, vpy);
        yp_grad = target.GradientLogDensity(yp);

        // B half-step
        vpy += 0.5 * delta * yp_grad;
        double minus_KE_y_final = minus_kinetic_energy(vpy, Sigma_chol_u);

        // Metropolis correction
        double logpi_yp = target.LogDensity(yp);
        double logHR_y = logpi_yp + minus_KE_y_final - logpi_y - minus_KE_y_init;
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_y > 0 || log_u < logHR_y) // Accept
        {
            acceptances_y[it - 1] = true;
            vy = vpy;
            y = yp;
            y_grad = yp_grad;
            logpi_y = logpi_yp;
        }
        else // Turn around
        {
            vy = (-vy).eval();
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }
}

/************* Coupled MCMC **********/

// SGLD vs MALA, CRN coupling
template <typename m1, typename m2>
void sgld_mala_CRN(const TargetTallData &target,
                   const int &batch_size,
                   const Eigen::Ref<const Eigen::VectorXd> &x0,    // SGLD
                   const Eigen::Ref<const Eigen::VectorXd> &y0,    // MALA
                   const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // MALA proposal covariance == SGLD volatility increment
                   const int &iter,
                   const int &thin,
                   Eigen::MatrixBase<m1> &xs,
                   Eigen::MatrixBase<m2> &ys,
                   double &acceptance_rate_y)
{
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma); // For proposal generation

    // SGLD
    VectorXd x = x0;
    xs.row(0) = x;

    // MALA
    VectorXd y = y0;
    VectorXd y_grad = target.GradientLogDensity(y);
    VectorXd y_mean = y + 0.5 * precondition(Sigma, y_grad);
    double logpi_y = target.LogDensity(y);
    ys.row(0) = y;
    int accepted_y = 0;

    for (int it = 1; it <= iter; it++)
    {
        VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));

        // X: SGLD
        VectorXd x_grad_potential = subsampled_gradpotential_noreplace(x, target, batch_size);
        x = x - 0.5 * precondition(Sigma, x_grad_potential) + Lz;

        // Y: MALA
        VectorXd yp = y_mean + Lz;
        VectorXd yp_grad = target.GradientLogDensity(yp);
        VectorXd yp_mean = yp + 0.5 * precondition(Sigma, yp_grad);
        double logpi_yp = target.LogDensity(yp);

        double logHR_y = logpi_yp - logpi_y + mala_proposal_logdensity_difference(y, y_mean, y_grad, yp, yp_mean, yp_grad);
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_y > 0 || log_u < logHR_y)
        {
            ++accepted_y;
            y = yp;
            y_grad = yp_grad;
            y_mean = yp_mean;
            logpi_y = logpi_yp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }

    acceptance_rate_y = (double)accepted_y / (double)iter;
}

// SGLDcv vs MALA, CRN coupling
template <typename m1, typename m2>
void sgldcv_mala_CRN(const TargetTallData &target,
                     const Eigen::Ref<const Eigen::VectorXd> &mode,
                     const Eigen::Ref<const Eigen::VectorXd> &gradient_at_mode,
                     const int &batch_size,
                     const Eigen::Ref<const Eigen::VectorXd> &x0,    // SGLD
                     const Eigen::Ref<const Eigen::VectorXd> &y0,    // MALA
                     const Eigen::Ref<const Eigen::MatrixXd> &Sigma, // MALA proposal covariance == SGLD volatility increment
                     const int &iter,
                     const int &thin,
                     Eigen::MatrixBase<m1> &xs,
                     Eigen::MatrixBase<m2> &ys,
                     double &acceptance_rate_y)
{
    int n = target.Count();
    int d = x0.size();
    MatrixXd Sigma_chol_l = chol_l(Sigma); // For proposal generation

    // X: SGLD
    vector<VectorXd> gradients_at_mode(n);
    for (int i = 0; i < n; i++)
        gradients_at_mode[i] = target.GradientPotential(i, mode);
    VectorXd x = x0;
    xs.row(0) = x;

    // Y: MALA
    VectorXd y = y0;
    VectorXd y_grad = target.GradientLogDensity(y);
    VectorXd y_mean = y + 0.5 * precondition(Sigma, y_grad);
    double logpi_y = target.LogDensity(y);
    int accepted_y = 0;
    ys.row(0) = y;

    for (int it = 1; it <= iter; it++)
    {
        VectorXd Lz = precondition_lowertri(Sigma_chol_l, RNG::GaussianVector(d));

        // X: SGLDcv
        VectorXd x_grad_potential = subsampled_gradpotential_noreplace_cv(x, target, batch_size, gradient_at_mode, gradients_at_mode);
        x = x - 0.5 * precondition(Sigma, x_grad_potential) + Lz;

        // Y: MALA
        VectorXd yp = y_mean + Lz;
        VectorXd yp_grad = target.GradientLogDensity(yp);
        VectorXd yp_mean = yp + 0.5 * precondition(Sigma, yp_grad);
        double logpi_yp = target.LogDensity(yp);

        double logHR_y = logpi_yp - logpi_y + mala_proposal_logdensity_difference(y, y_mean, y_grad, yp, yp_mean, yp_grad);
        double log_u = log(RNG::runif(RNG::rng));
        if (logHR_y > 0 || log_u < logHR_y)
        {
            ++accepted_y;
            y = yp;
            y_grad = yp_grad;
            y_mean = yp_mean;
            logpi_y = logpi_yp;
        }

        if (it % thin == 0)
            xs.row(it / thin) = x;
        if (it % thin == 0)
            ys.row(it / thin) = y;
    }

    acceptance_rate_y = (double)accepted_y / (double)iter;
}
