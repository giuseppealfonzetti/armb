#ifndef utils_H
#define utils_H

#include <random>
#include <span>

//' Randomly shuffle rows of a matrix
//'
//' @param X matrix
//' @param SEED seed for rng
//'
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd shuffleRows(
  const Eigen::MatrixXd& X,
  const int SEED
){
  std::mt19937 rng(SEED);
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shufX(X.rows());
  shufX.setIdentity();
  std::shuffle(shufX.indices().data(), shufX.indices().data()+shufX.indices().size(), rng);
  return shufX*X;
}

//' Randomly shuffle elements of a vector
//'
//' @param X vector
//' @param SEED seed for rng
//'
//' @export
// [[Rcpp::export]]
Eigen::VectorXd shuffleVec(
  const Eigen::VectorXd& X,
  const int SEED
){
  std::mt19937 rng(SEED);
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shufX(X.size());
  shufX.setIdentity();
  std::shuffle(shufX.indices().data(), shufX.indices().data()+shufX.indices().size(), rng);
  return shufX*X;
}

//' Resample sequence
//'
//' Resample with replacement the sequence from 0 to N-1
//'
//' @param N length of the sequence
//' @param SEED seed for rng
//'
//' @export
// [[Rcpp::export]]
std::vector<int> resampleN(
  const int N,
  const int SEED
){
  std::mt19937 rng(SEED);
  std::uniform_int_distribution<> resample(0, N - 1);
  std::vector<int> out(N);
  std::generate(out.begin(), out.end(), [&resample, &rng]() { return resample(rng); });
  return out;
}

//' Resample sequence
//'
//' Resample with replacement the sequence from 0 to N-1
//'
//' @param N length of the sequence
//' @param SEED seed for rng
//'
//' @export
// [[Rcpp::export]]
Eigen::VectorXd sliceVec(
  const std::vector<int> SLICE_IDX,
  const Eigen::VectorXd X
){

  return X(SLICE_IDX);
}

//' Resample sequence
//'
//' Resample with replacement the sequence from 0 to N-1
//'
//' @param N length of the sequence
//' @param SEED seed for rng
//'
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sliceMat(
  const std::vector<int> SLICE_IDX,
  const Eigen::MatrixXd X
){

  return X(SLICE_IDX, Eigen::all);
}

//' @export
// [[Rcpp::export]]
std::vector<int> subsetIVec(
  const std::vector<int> &X,
  const int START,
  const int LEN
){
  std::vector<int> subvector = {X.begin() + START, X.begin() + START + LEN};
  return subvector;
}

//' @export
// [[Rcpp::export]]
std::vector<int> shuffleIVec(
  std::vector<int> &X,
  const int SEED
){
  std::mt19937 rng(SEED);
  std::shuffle(std::begin(X), std::end(X), rng);
  return X;
}
#endif
