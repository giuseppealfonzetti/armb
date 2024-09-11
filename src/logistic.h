#ifndef logistic_H
#define logistic_H

namespace logistic
{
  //' Compute b(eta) for logistic response
  //' b(eta) = log(1+exp(eta))
  Eigen::VectorXd b(const Eigen::VectorXd& ETA,
                    const int N){
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
}

#endif
