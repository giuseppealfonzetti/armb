#ifndef response_H
#define response_H
#include <float.h>
#define DOUBLE_EPS DBL_EPSILON
static const double THRESH = 30.;
static const double MTHRESH = -30.;
static const double INVEPS = 1/DOUBLE_EPS;
static const double QNORM_EPS = R::qnorm(DOUBLE_EPS, 0, 1, true, false);

namespace family{

  // Binomial family
  namespace binomial{
    double nll_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU){
      return -(MU.array().log() * Y.array() + (1-Y.array())*(-MU).array().log1p()).sum();}

    double dev_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU){
          return -2*(MU.array().log() * Y.array() + (1-Y.array())*(-MU).array().log1p()).sum();}

  }

  // Poisson family
  namespace poisson{
    double nll_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU){
      return (MU.array()-Y.array()*MU.array().max(DOUBLE_EPS).log()).sum();
      }

    double dev_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU){
      Eigen::VectorXd log_term = (Y.array() > 0).select(Y.array() * (Y.array() / MU.array()).log(), 0);
      return 2 * (log_term.array() - (Y - MU).array()).sum();
      }

  }
}

namespace link{

  namespace logit{
    Eigen::VectorXd linkfun_(const Eigen::Ref<const Eigen::VectorXd> MU){return MU.array().log()-(-MU).array().log1p();}
    Eigen::VectorXd linkinv_(const Eigen::Ref<const Eigen::VectorXd> ETA){return 1/( 1+ETA.unaryExpr([](double etaij){ return (-etaij < MTHRESH) ? DOUBLE_EPS :((-etaij > THRESH) ? INVEPS : exp(-etaij)); }).array());}
    Eigen::VectorXd dmudeta_(const Eigen::Ref<const Eigen::VectorXd> ETA){
      return linkinv_(ETA).unaryExpr([](double mui){return mui-pow(mui, 2);});
      }
  }

  namespace log{
    Eigen::VectorXd linkfun_(const Eigen::Ref<const Eigen::VectorXd> MU){return MU.array().log();}
    Eigen::VectorXd linkinv_(const Eigen::Ref<const Eigen::VectorXd> ETA){
      return ETA.array().exp().max(DOUBLE_EPS);
      }
    Eigen::VectorXd dmudeta_(const Eigen::Ref<const Eigen::VectorXd> ETA){return ETA.array().exp();}
  }

}

class Response{
private:
  std::string family_lab;
  std::string link_lab;
  std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> fun_linkfun;
  std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> fun_linkinv;
  std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> fun_dmudeta;
  std::function<double(const Eigen::Ref<const Eigen::VectorXd>, const Eigen::Ref<const Eigen::VectorXd>)> fun_nll;
  std::function<double(const Eigen::Ref<const Eigen::VectorXd>, const Eigen::Ref<const Eigen::VectorXd>)> fun_dev;

public:
  bool canonical_flag = false;
  Response(std::string FAMILY_LAB_, std::string LINK_LAB_): family_lab(FAMILY_LAB_), link_lab(LINK_LAB_){
    // Family dispatching
    if(FAMILY_LAB_=="binomial"){
      fun_nll = &family::binomial::nll_;
      fun_dev = &family::binomial::dev_;
    }else if(FAMILY_LAB_=="poisson"){
      fun_nll = &family::poisson::nll_;
      fun_dev = &family::poisson::dev_;
    }else{Rcpp::stop("Family not supported");}

    // Link dispatching
    if(LINK_LAB_=="logit"){
      fun_linkinv = &link::logit::linkinv_;
      fun_linkfun = &link::logit::linkfun_;
      fun_dmudeta = &link::logit::dmudeta_;
    }else if(LINK_LAB_=="log"){
      fun_linkinv = &link::log::linkinv_;
      fun_linkfun = &link::log::linkfun_;
      fun_dmudeta = &link::log::dmudeta_;
    }else{Rcpp::stop("Link not supported");}

  }

  // functions to be dispatched according to family and link
  Eigen::VectorXd linkinv_(const Eigen::Ref<const Eigen::VectorXd> ETA);
  Eigen::VectorXd linkfun_(const Eigen::Ref<const Eigen::VectorXd> MU);
  Eigen::VectorXd dmudeta_(const Eigen::Ref<const Eigen::VectorXd> ETA);

  double nll_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU);
  double dev_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU);

};

Eigen::VectorXd Response::linkinv_(const Eigen::Ref<const Eigen::VectorXd> ETA){return this->fun_linkinv(ETA);}
Eigen::VectorXd Response::linkfun_(const Eigen::Ref<const Eigen::VectorXd> MU){return this->fun_linkfun(MU);}
Eigen::VectorXd Response::dmudeta_(const Eigen::Ref<const Eigen::VectorXd> ETA){return this->fun_dmudeta(ETA);}

double Response::nll_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU){return this->fun_nll(Y, MU);}
double Response::dev_(const Eigen::Ref<const Eigen::VectorXd> Y, const Eigen::Ref<const Eigen::VectorXd> MU){return this->fun_dev(Y, MU);}


#endif
