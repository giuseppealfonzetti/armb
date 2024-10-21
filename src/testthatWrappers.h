#ifndef testhatWrappers_H
#define testhatWrappers_H
#include "response.h"
#include "utils.h"

//' Compute nll and ngr for logistic regression
//'
//' @description Used for internal testing
//'
//' @param Y response
//' @param X design matrix
//' @param THETA parameter vector
//'
//' @returns Provides a list with the negative log likelihood and its gradient
//'
//' @export
// [[Rcpp::export]]
Rcpp::List test_glm(
    const Eigen::Map<Eigen::VectorXd> Y,
    const Eigen::Map<Eigen::MatrixXd> X,
    std::string FAMILY,
    std::string LINK,
    const Eigen::Map<Eigen::VectorXd> THETA
){

  Response resp(FAMILY, LINK);

  const Eigen::VectorXd mu = resp.linkinv_(X*THETA);
  const int n = Y.size();

  const double nll = resp.nll_(Y, mu)/n;
  Eigen::VectorXd ngr = X.transpose()*(mu-Y)/n;
  const double dev = resp.dev_(Y, mu);


  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("mu") = mu,
      Rcpp::Named("nll") = nll,
      Rcpp::Named("ngr") = ngr,
      Rcpp::Named("dev") = dev
    );

  return output;
}

#endif
