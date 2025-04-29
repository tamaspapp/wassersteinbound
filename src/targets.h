#include <Eigen/Core>
#include <Eigen/SparseCore>

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef const Eigen::Ref<const MatrixXd> constRefMat;
typedef const Eigen::Ref<const VectorXd> constRefVec;

typedef Eigen::SparseMatrix<double> SparseMat;
typedef const Eigen::Ref<const Eigen::SparseMatrix<double>> constSparseRefMat;

// Generic target class
class Target
{
public:
    virtual ~Target(){};

    virtual double LogDensity(constRefVec &x) const = 0;

    virtual VectorXd GradientLogDensity(constRefVec &x) const = 0;
};

class StochasticVolatility : public Target
{
private:
    VectorXd y_sq;
    double beta_;
    double sigma_;
    double phi_;
    int d_;

public:
    StochasticVolatility(constRefVec &y, const double &beta, const double &sigma, const double &phi)
        : beta_(beta), sigma_(sigma), phi_(phi)
    {
        y_sq = y.array().square();
        d_ = y_sq.size();
    }

    virtual double LogDensity(constRefVec &x) const
    {
        VectorXd exp_minus_x(d_); exp_minus_x = (-x).array().exp(); // Need to do this to avoid issues in the dot product below

        double neg2ll = x.sum();
        neg2ll += y_sq.dot(exp_minus_x) / (beta_ * beta_);
        neg2ll += (phi_ * x.head(d_ - 1) - x.tail(d_ - 1)).squaredNorm() / (sigma_ * sigma_);
        neg2ll += (1. - phi_ * phi_) * x(0) * x(0) / (sigma_ * sigma_);
        double ll = (-0.5) * neg2ll;
        return ll;
    }

    virtual VectorXd GradientLogDensity(constRefVec &x) const
    {
        VectorXd exp_minus_x(d_); exp_minus_x = (-x).array().exp(); // Need to do this to avoid issues, see above.

        VectorXd grad = VectorXd::Constant(d_, -0.5);
        grad += (0.5 / (beta_ * beta_)) * y_sq.cwiseProduct(exp_minus_x);
        grad(0) -= ((1 - phi_ * phi_)/ (sigma_ * sigma_)) * x(0);
        grad.head(d_ - 1) -= (phi_ / (sigma_ * sigma_)) * (phi_ * x.head(d_ - 1) - x.tail(d_ - 1));
        grad.tail(d_ - 1) += (phi_ * x.head(d_ - 1) - x.tail(d_ - 1)) / (sigma_ * sigma_);
        return grad;
    }
};

// pi(x) ~ e^(-||x||)/ (1 + e^(-||x||))^2
class MultivariateLogistic : public Target
{
private:
    int d_;

    virtual double transf_density(const double &t) const // Will only apply to non-negative t
    {
        return -(t + 2. * log1p(exp(-t)));
    } 

    virtual double transf_grad(const double &t) const // Will only apply to non-negative t
    {
        return expm1(-t) / (1. + exp(-t));
    }

public:
    MultivariateLogistic()
    {
    }

    virtual double LogDensity(constRefVec &x) const  // - ||x|| - 2 log(1 + exp(-||x||))
    {
        return transf_density(x.norm());
    }

    virtual VectorXd GradientLogDensity(constRefVec &x) const //  [ - 1 + 2 exp(-||x||)/{1 + exp(-||x||)} ] * (x / ||x||)
    {
        return transf_grad(x.norm()) * x.normalized();
    }
};

class Gaussian : public Target
{
private:
    constRefVec mu_;
    constRefMat Omega_;
    MatrixXd Omega_chol_u;

public:
    Gaussian(constRefVec &mu, constRefMat &Omega)
        : mu_(mu), Omega_(Omega)
    {
        Omega_chol_u = Omega_.llt().matrixU();
    }

    virtual double LogDensity(constRefVec &x) const
    {
        return - 0.5 * (Omega_chol_u.triangularView<Eigen::Upper>() * (x - mu_)).squaredNorm();
    }

    virtual VectorXd GradientLogDensity(constRefVec &x) const
    {
        return Omega_ * (mu_ - x);
    }
};

class SparseGaussian : public Target
{
private:
    constRefVec mu_;
    constSparseRefMat Omega_;

public:
    SparseGaussian(constRefVec &mu, constSparseRefMat &Omega)
        : mu_(mu), Omega_(Omega)
    {
    }
    
    virtual double LogDensity(constRefVec &x) const
    {
        return - 0.5 * (x - mu_).dot(Omega_ * (x - mu_));
    }

    virtual VectorXd GradientLogDensity(constRefVec &x) const
    {
        return Omega_ * (mu_ - x);
    }
};

using Eigen::ArrayXd;
typedef const Eigen::Ref<const Eigen::ArrayXd> constRefArray;

class OneDimGaussMixture : public Target
{
private:
    ArrayXd log_p_;
    constRefArray mu_;
    ArrayXd inv_sigma_;
    ArrayXd log_inv_sigma_;

    virtual double logsumexp(constRefArray &x) const
    {
        double max = x.maxCoeff();
        return max + log((x - max).exp().sum());
    }

public:
    OneDimGaussMixture(constRefArray &p, constRefArray &mu, constRefArray &sigma)
        : mu_(mu)
    {
        log_p_ = p.log();
        inv_sigma_ = sigma.inverse();
        log_inv_sigma_ = inv_sigma_.log();
    }

    virtual double LogDensity(constRefVec &x) const
    {
        ArrayXd exponents = log_p_ + log_inv_sigma_ - 0.5 * ((x(0) - mu_) * inv_sigma_).square();
        return logsumexp(exponents);
    }

    virtual VectorXd GradientLogDensity(constRefVec &x) const
    {
        ArrayXd exponents_down = log_p_ + log_inv_sigma_ - 0.5 * ((x(0) - mu_) * inv_sigma_).square();
        ArrayXd exponents_up = exponents_down + 2 * log_inv_sigma_;
        double max_down = exponents_down.maxCoeff();
        double max_up = exponents_up.maxCoeff();
        
        VectorXd rv(x.size()); // x.size() = 1 since this is one-dimensional
        rv(0) = exp(max_up - max_down) * ((exponents_up - max_up).exp() * (mu_ - x(0))).sum() / (exponents_down - max_down).exp().sum(); // Hopefully more stable
        return rv; 
    }
};

/****************** Tall data targets *******************/

// Generic big data target class
class TargetTallData : public Target
{
public:
    virtual ~TargetTallData(){};

    virtual int Count() const = 0;
    virtual int Dimension() const = 0;

    virtual int64_t NumPotentialEvaluations() const = 0;

    virtual double Potential(constRefVec &theta) const = 0;
    virtual double Potential(const int &n, constRefVec &theta) const = 0;
    virtual double LogDensity(constRefVec &theta) const
    {
        return -Potential(theta);
    }

    virtual VectorXd GradientPotential(constRefVec &theta) const = 0;
    virtual VectorXd GradientPotential(const int &n, constRefVec &theta) const = 0;
    virtual VectorXd GradientLogDensity(constRefVec &theta) const
    {
        return -GradientPotential(theta);
    }

    virtual MatrixXd HessianPotential(constRefVec &theta) const = 0;
    virtual MatrixXd HessianPotential(const int &n, constRefVec &theta) const = 0;
    virtual MatrixXd HessianLogDensity(constRefVec &theta) const
    {
        return -HessianPotential(theta);
    }
};

// Logistic regression target
class LogisticRegression : public TargetTallData
{
private:
    constRefMat yX_;
    constRefVec lambda_;
    mutable int64_t num_potential_evaluations_;

    virtual double minuslogcdf(const double &t) const // minus log-sigmoid
    {
        if (t < -33.3)
        {
            return -t;
        }
        else if (t <= -18)
        {
            return exp(t) - t;
        }
        else if (t <= 37)
        {
            return log1p(exp(-t));
        }
        else
        {
            return exp(-t);
        }
    }

    virtual double minuslogcdfdash(const double &t) const // minus log-sigmoid dashed
    {
        if (t < 0)
        {
            return - 1. / (1. + exp(t));
        }
        else
        {
            return - exp(-t) / (1. + exp(-t));
        }
    }

    virtual double minuslogcdfdoubledash(const double &t) const // minus log-sigmoid double-dashed
    {
        if (t < 0)
        {
            return exp(t) / pow(1. + exp(t), 2);
        }
        else
        {
            return exp(-t) / pow(1. + exp(-t), 2);
        }
    }

public:
    LogisticRegression(constRefMat &yX, constRefVec &lambda)
        : yX_(yX), lambda_(lambda), num_potential_evaluations_(0)
    {
    }

    virtual int Count() const
    {
        return yX_.rows();
    }

    virtual int Dimension() const
    {
        return yX_.cols();
    }

    virtual int64_t NumPotentialEvaluations() const
    {
        return num_potential_evaluations_;
    }

    virtual double Potential(const int &n, constRefVec &theta) const
    {
        num_potential_evaluations_ += 1;
        return minuslogcdf(yX_.row(n).dot(theta)) + 0.5 * (theta.array() / lambda_.array()).matrix().squaredNorm() / yX_.rows();
    }
    virtual double Potential(constRefVec &theta) const
    {
        double nll = 0.0;
        for (int n = 0; n < yX_.rows(); n++)
            nll += Potential(n, theta);
        return nll;
    }

    virtual VectorXd GradientPotential(const int &n, constRefVec &theta) const
    {
        double coeff = minuslogcdfdash(yX_.row(n).dot(theta));
        VectorXd grad_n  = coeff * yX_.row(n);
        grad_n.array() += (theta.array() / lambda_.array().square()) / yX_.rows();
        return grad_n;
    }
    virtual VectorXd GradientPotential(constRefVec &theta) const
    {
        VectorXd rv = VectorXd::Zero(theta.size());
        for (int n = 0; n < yX_.rows(); n++)
            rv += GradientPotential(n, theta);
        return rv;
    }

    virtual MatrixXd HessianPotential(const int &n, constRefVec &theta) const
    {
        double coeff = minuslogcdfdoubledash(yX_.row(n).dot(theta));
        MatrixXd hess_n = (coeff * yX_.row(n).transpose()) * yX_.row(n);
        hess_n.diagonal().array() += lambda_.array().square() / yX_.rows();
        return hess_n;
    }
    virtual MatrixXd HessianPotential(constRefVec &theta) const
    {
        MatrixXd rv = MatrixXd::Zero(yX_.cols(), yX_.cols());
        for (int n = 0; n < yX_.rows(); n++)
            rv += HessianPotential(n, theta);
        return rv;
    }
};
