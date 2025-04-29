#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using Eigen::Ref;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXi;
using Eigen::ArrayXXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::PermutationMatrix;
using Eigen::Dynamic;

// ------- Rank samples column-wise ------- //
void rank_inplace(Ref<ArrayXi> rank, const Ref<const ArrayXd> x){
  int n = x.size();
  
  // Get order
  ArrayXi order = ArrayXi::LinSpaced(n, 0, n-1); // Arguments for LinSpaced: (size, low, high)
  std::iota(order.data(), order.data() + n, 0); // "array.data()" points to first entry of "array"
  std::sort(order.data(), order.data() + n, [&x](int i, int j){return x[i] < x[j];});
  
  // Get rank from order
  for(int i = 0; i < order.size(); i++)
    rank(order(i)) = i;
}

//'@export
//[[Rcpp::export(rng = false)]]
Eigen::ArrayXXi colwise_rank_cpp(const Eigen::Map<Eigen::ArrayXXd> &mat){
    ArrayXXi ranks(mat.rows(), mat.cols());
    for(int i = 0; i < mat.cols(); i++)
        rank_inplace(ranks.col(i), mat.col(i));
    return ranks;
}


// ------- Permute column-wise ------- //

//'@export
//[[Rcpp::export(rng = false)]]
void permute_matrix_colwise_inplace_cpp(Eigen::Map<Eigen::MatrixXd> &mat, 
                                        const Eigen::Map<Eigen::MatrixXi> &idx_colwise){

    for(int i = 0; i < mat.cols(); i++){
        PermutationMatrix<Dynamic, Dynamic> perm;
        perm.indices() = idx_colwise.col(i);
        mat.col(i) = perm * mat.col(i);
    }
}
// If "idx_colwise" holds the column-wise ranks of the samples, then this will sort the matrix "m" column-wise.


// ------- Column-wise squared norms between matrices "x" and "y", after entries indexed by "rm_idx_x" and "rm_idx_y" in the respective columns are removed ------- //

// Equivalent R code, up to 0-indexing: sum((x[-rm_idx_x] - y[-rm_idx_y])^2)
double squared_norm_with_skip(const Ref<const VectorXd> x, // Assumes that x.size() == y.size()
                              const Ref<const VectorXd> y,
                              Ref<VectorXi> rm_idx_x,
                              Ref<VectorXi> rm_idx_y){
    double squaredist = 0.0;
    int n = x.size();
    int skip_n = rm_idx_x.size();
    
    std::sort(rm_idx_x.begin(), rm_idx_x.end());
    std::sort(rm_idx_y.begin(), rm_idx_y.end());

    int skip_counter_x = 0;
    int skip_counter_y = 0;

    int jx = 0;
    int jy = 0;
    while(true)
    {
        while(skip_counter_x < skip_n && jx == rm_idx_x[skip_counter_x]) // Skip consecutive entries
        {
            jx++;
            skip_counter_x++;
        }
        while(skip_counter_y < skip_n && jy == rm_idx_y[skip_counter_y])
        {
            jy++;
            skip_counter_y++;
        }
        if(jx >= n || jy >= n) break; // We're finished: nothing more to sum over

        double d = x[jx] - y[jy];
        squaredist += d*d;
        jx++; jy++;
    }
    return squaredist;
}

//'@export
// [[Rcpp::export(rng = false)]]
Eigen::VectorXd colwise_squared_norm_with_skip_cpp(const Eigen::Map<Eigen::MatrixXd> &x,
                                                   const Eigen::Map<Eigen::MatrixXd> &y, 
                                                   Eigen::Map<Eigen::MatrixXi> &rm_idx_x,
                                                   Eigen::Map<Eigen::MatrixXi> &rm_idx_y)
{
    int n = x.cols();
    Eigen::VectorXd squaredist(n);
    for(int i = 0; i < n; i++)
        squaredist(i) = squared_norm_with_skip(x.col(i), y.col(i), rm_idx_x.col(i), rm_idx_y.col(i));

    return squaredist;
}

