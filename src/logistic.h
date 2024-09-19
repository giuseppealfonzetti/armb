#ifndef logistic_H
#define logistic_H

namespace logistic
{
  //' Compute b(eta) for logistic response
  //' b(eta) = log(1+exp(eta))
  Eigen::VectorXd b(
      const Eigen::VectorXd& ETA,
      const int N
      ){
    Eigen::VectorXd b(N);
    for(int i = 0; i < N; i++){
      b[i] = log1pexp(ETA[i]);
    }
    return b;
  }

  //' Compute mean function for logistic response
  //' mu = exp(eta - b(eta))
  Eigen::VectorXd mu(const Eigen::VectorXd& ETA,
                     const Eigen::VectorXd& B){
    Eigen::VectorXd mu = (ETA - B).array().exp();
    return mu;
  }

  //' Compute nll for logistic regression
  double nll(
    const Eigen::VectorXd& Y,
    const Eigen::VectorXd& ETA,
    const int N
  ){
    double yeta = Y.dot(ETA);
    Eigen::VectorXd b = logistic::b(ETA, N);
    double nll = b.sum() - yeta;
    return nll;
  }

  //' Compute negative gradient for logistic regression
  Eigen::VectorXd ngr(
    const Eigen::VectorXd& Y,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& ETA,
    const int N
  ){
    const int m = X.cols();                     // Number of parameters
    Eigen::VectorXd nres = logistic::b(ETA, N); // Initialise vector of negative residuals computing b(eta) = log(1+exp(eta))
    nres = logistic::mu(ETA, nres);             // Compute the mean function mu = exp(eta - b(eta))
    nres -= Y;                                  // Compute negative residuals = mu - Y;

    Eigen::VectorXd ngr = X.transpose()*nres;   // Compute the gradient
    return(ngr);
  }

}

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
Rcpp::List logistic_wrapper(
    const Eigen::Map<Eigen::VectorXd> Y,
    const Eigen::Map<Eigen::MatrixXd> X,
    const Eigen::Map<Eigen::VectorXd> THETA
){

  const Eigen::VectorXd eta = X*THETA;
  const int n = Y.size();

  const double nll = logistic::nll(Y, eta, n);
  const Eigen::VectorXd ngr = logistic::ngr(Y, X, eta, n);

  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("nll") = nll,
      Rcpp::Named("ngr") = ngr
    );

  return output;
}
#endif
