#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;
using namespace Rcpp;


// [[Rcpp::export]]
Eigen::MatrixXd dense_dense_interaction_kronecker(
    Eigen::MatrixXd X,
    Eigen::MatrixXd Y
  ) {
  VectorXd ones_x = VectorXd::Constant(Y.rows(), 1);
  VectorXd ones_y = VectorXd::Constant(X.rows(), 1);
  return kroneckerProduct(ones_x, X).array() * kroneckerProduct(Y, ones_y).array();
}


// [[Rcpp::export]]
Eigen::MatrixXd dense_sparse_interaction_kronecker(
    Eigen::MatrixXd X,
    Eigen::SparseMatrix<double> Y
) {
  
  MatrixXd output(X.rows() * Y.rows(), X.cols());
  SparseMatrix<double> Y_transpose = Y.transpose();
  for (int k=0; k<Y_transpose.outerSize(); ++k) {
    MatrixXd X_scaled = X * VectorXd(Y_transpose.col(k)).asDiagonal();
    output.middleRows(k * X.rows(), X.rows()) = X_scaled;
  }
  return output;
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double> sparse_dense_interaction_kronecker(
    Eigen::SparseMatrix<double> X,
    Eigen::MatrixXd Y
) {
  
  SparseMatrix<double, RowMajor> output(X.rows() * Y.rows(), X.cols());
  MatrixXd Y_transpose = Y.transpose();
  for (int k=0; k<Y_transpose.outerSize(); ++k) {
    SparseMatrix<double> X_scaled = X * VectorXd(Y_transpose.col(k)).asDiagonal();
    X_scaled.prune(0.0);
    output.middleRows(k * X.rows(), X.rows()) = X_scaled;
  }
  return output;
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double> sparse_sparse_interaction_kronecker(
    Eigen::SparseMatrix<double> X,
    Eigen::SparseMatrix<double> Y
  ) {

  SparseMatrix<double, RowMajor> output(X.rows() * Y.rows(), X.cols());
  SparseMatrix<double> Y_transpose = Y.transpose();
  for (int k=0; k<Y_transpose.outerSize(); ++k) {
    SparseMatrix<double> X_scaled = X * VectorXd(Y_transpose.col(k)).asDiagonal();
    X_scaled.prune(0.0);
    output.middleRows(k * X.rows(), X.rows()) = X_scaled;
  }
  return output;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> RcppSparseMatrixRbindList(Rcpp::List matrices) {
  int total_columns = 0;
  int total_rows = 0;
  int nnz = 0;
  // Figure out how many rows are there total, so we can preallocate a big matrix.
  // Also figure out how many nonzeros to reserve.
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::SparseMatrix<double>> m = matrices[i];
    int nrow = m.rows();
    total_rows += nrow;
    if (i == 0) {
      total_columns = m.cols(); // since this is rbind, all matrices have the same ncols
    }
    nnz += m.nonZeros();
  }
  
  SparseMatrix<double, RowMajor> out(total_rows, total_columns);
  out.reserve(nnz);
  int startRow = 0;
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::SparseMatrix<double>> m = matrices[i];
    int nrow = m.rows();
    out.middleRows(startRow, nrow) = m;
    startRow += nrow;
  }
  
  return out;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> RcppSparseMatrixCbindList(Rcpp::List matrices) {
  int total_columns = 0;
  int total_rows = 0;
  int nnz = 0;
  // Figure out how many columns are there total, so we can preallocate a big matrix.
  // Also figure out how many nonzeros to reserve.
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::SparseMatrix<double>> m = matrices[i];
    int ncols = m.cols();
    total_columns += ncols;
    if (i == 0) {
      total_rows = m.cols(); // since this is cbind, all matrices have the same nrows
    }
    nnz += m.nonZeros();
  }
  
  SparseMatrix<double> out(total_rows, total_columns);
  out.reserve(nnz);
  int startCol = 0;
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::SparseMatrix<double>> m = matrices[i];
    int ncol = m.cols();
    out.middleCols(startCol, ncol) = m;
    startCol += ncol;
  }
  
  return out;
}

// [[Rcpp::export]]
Eigen::MatrixXd RcppMatrixRbindList(Rcpp::List matrices) {
  int total_columns = 0;
  int total_rows = 0;
  // Figure out how many rows are there total, so we can preallocate a big matrix.
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::SparseMatrix<double>> m = matrices[i];
    int nrow = m.rows();
    total_rows += nrow;
    if (i == 0) {
      total_columns = m.cols(); // since this is rbind, all matrices have the same ncols
    }
  }
  
  Eigen::MatrixXd out(total_rows, total_columns);
  int startRow = 0;
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::SparseMatrix<double>> m = matrices[i];
    int nrow = m.rows();
    out.middleRows(startRow, nrow) = m;
    startRow += nrow;
  }
  
  return out;
}

// [[Rcpp::export]]
Eigen::MatrixXd RcppMatrixCbindList(Rcpp::List matrices) {
  int total_columns = 0;
  int total_rows = 0;
  // Figure out how many columns are there total, so we can preallocate a big matrix.
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::MatrixXd> m = matrices[i];
    int ncols = m.cols();
    total_columns += ncols;
    if (i == 0) {
      total_rows = m.cols(); // since this is cbind, all matrices have the same nrows
    }
  }
  
  Eigen::MatrixXd out(total_rows, total_columns);
  int startCol = 0;
  for (int i = 0; i < matrices.length(); ++i) {
    Eigen::Map<Eigen::MatrixXd> m = matrices[i];
    int ncol = m.cols();
    out.middleCols(startCol, ncol) = m;
    startCol += ncol;
  }
  
  return out;
}