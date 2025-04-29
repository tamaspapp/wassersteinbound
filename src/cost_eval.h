#include <Eigen/Core>

typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> RowMajorMat;
Eigen::MatrixXd squared_euclidean_cost_matrix(const Eigen::Ref<const Eigen::MatrixXd> &x, const Eigen::Ref<const Eigen::MatrixXd> &y);
RowMajorMat squared_euclidean_cost_matrix_rowmajor(const Eigen::Ref<const Eigen::MatrixXd> &x, const Eigen::Ref<const Eigen::MatrixXd> &y);
//Eigen::MatrixXf squared_euclidean_cost_matrix_float(const Eigen::Ref<const Eigen::MatrixXf> &x, const Eigen::Ref<const Eigen::MatrixXf> &y);
