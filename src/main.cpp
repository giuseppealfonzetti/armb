#include <Rcpp.h>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
#include <RcppClock.h>
#include "logistic.h"
#include "response.h"
#include "utils.h"
#include "testthatWrappers.h"


//' Fit logistic regression via ARM
//'
//' @param Y response
//' @param X design matrix
//' @param THETA0 starting values
//' @param MAXT max number of iterations
//' @param MAXE max number of epochs
//' @param BURN burn-in period
//' @param BATCH mini-natch dimension
//' @param STEPSIZE0 initial stepsize
//' @param PAR1 par1
//' @param PAR2 par2
//' @param PAR3 par3
//' @param SKIP_PRINT How many iterations to skip before printing diagnostics
//' @param VERBOSE verbose output
//'
//' @export
// [[Rcpp::export]]
Rcpp::List armLR2(
  const Eigen::Map<Eigen::VectorXd> Y,
  const Eigen::Map<Eigen::MatrixXd> X,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int MAXT,
  const int BURN,
  const int BATCH,
  const double STEPSIZE0,
  const double PAR1,
  const double PAR2,
  const double PAR3,
  const int STORE,
  const int SKIP_PRINT,
  const int SEED,
  const bool VERBOSE
){

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("main");

  // Identify dimensions
  const int n = Y.size();
  const int m = THETA0.size();
  const int store = std::max(double(STORE), 2.0);
  const int skip_mem = MAXT / (store-1);

  std::vector<Eigen::VectorXd> theta_path;
  std::vector<int> iters_path;
  iters_path.push_back(0);
  theta_path.push_back(THETA0);

  std::vector<double> nll;
  Eigen::VectorXd theta = THETA0;
  Eigen::VectorXd av_theta = THETA0;
  int last_update = 0;


  Eigen::MatrixXd x = X;
  Eigen::VectorXd y = Y;
  int idx = 0;
  int shf = 0;
  int stored = 2;
  for(int t = 0; t < MAXT; t++){
    Rcpp::checkUserInterrupt();
    if((idx+BATCH)>n){
      idx = 0;
      Eigen::MatrixXd x = shuffleRows(X, SEED + shf);
      Eigen::VectorXd y = shuffleVec(Y, SEED + shf);
      shf++;
    }

    const Eigen::MatrixXd x_t = x(Eigen::seqN(idx, BATCH), Eigen::all);
    const Eigen::VectorXd y_t = y.segment(idx, BATCH);
    const Eigen::VectorXd eta_t = x_t*theta;
    const Eigen::VectorXd ngr = logistic::ngr(y_t, x_t, eta_t, BATCH)/BATCH;

    double stepsize_t = STEPSIZE0 * PAR1 * pow(1 + PAR2*STEPSIZE0*t, -PAR3);
    theta -= stepsize_t * ngr;

    if(VERBOSE & (t % SKIP_PRINT == 0)) Rcpp::Rcout  <<"Iter:"<<t<<"| Idxs:"<<idx<<"-"<<idx+BATCH <<"| norm:"<< std::setprecision(4) << ngr.norm() <<"\n";

    if(t <= BURN){
      av_theta = theta;
    }else{
      av_theta = ( (t - BURN) * av_theta + theta ) / (t - BURN + 1);
    }

    idx += BATCH;
    last_update += 1;
    if((t==MAXT-1) | ((((t + 1) % skip_mem) == 0)&( stored<store))){
      theta_path.push_back(av_theta);
      iters_path.push_back(t + 1);
      stored++;
    }
    // nll.push_back(logistic::nll(Y, X*av_theta, n)/n);
  }

  clock.tock("main");
  clock.stop("clock");

  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("skip_mem") = skip_mem,
    Rcpp::Named("theta_path") = theta_path,
    Rcpp::Named("iters_path") = iters_path,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("av_theta") = av_theta,
    Rcpp::Named("last_update") = last_update
  );

  return output;
}

//' tune logistic regression via ARM
//'
//' @param Y response
//' @param X design matrix
//' @param THETA0 starting values
//' @param MAXT max number of iterations
//' @param MAXE max number of epochs
//' @param BURN burn-in period
//' @param BATCH mini-natch dimension
//' @param STEPSIZE0 initial stepsize
//' @param SCALE multiplying factor
//' @param MAXA maximum number of attempts
//' @param PAR1 par1
//' @param PAR2 par2
//' @param PAR3 par3
//' @param SKIP_PRINT How many iterations to skip before printing diagnostics
//' @param VERBOSE verbose output
//'
//' @export
// [[Rcpp::export]]
Rcpp::List tune_armLR(
  const Eigen::Map<Eigen::VectorXd> Y,
  const Eigen::Map<Eigen::MatrixXd> X,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int MAXT,
  const int BURN,
  const int BATCH,
  const double STEPSIZE0,
  const double SCALE,
  const double MAXA,
  const double PAR1,
  const double PAR2,
  const double PAR3,
  const bool AUTO_STOP,
  const int SKIP_PRINT,
  const int SEED,
  const bool VERBOSE
){

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("main");

  // Identify dimensions
  const int n = Y.size();
  const int m = THETA0.size();
  const int maxt = std::min(double(MAXT), double(int(n / BATCH)));

  int last_update = 0;

  const Eigen::MatrixXd x = shuffleRows(X, SEED);
  std::vector<double> stepsizes;
  std::vector<double> nlls;

  double stepsize0 = STEPSIZE0;
  for(int a = 0; a < MAXA; a++){

    int idx = 0;
    if(VERBOSE) Rcpp::Rcout  <<"Stepsize:"<<std::setprecision(4)<<stepsize0<<" | ";

    Rcpp::List fit = armLR2(Y, X, THETA0, MAXT, BURN, BATCH, stepsize0,
                           PAR1, PAR2, PAR3, 1, 10, SEED, false);

    stepsizes.push_back(stepsize0);
    Eigen::VectorXd av_theta = fit["av_theta"];
    double nll = logistic::nll(Y, X*av_theta, n)/n;
    if(VERBOSE) Rcpp::Rcout  <<"nll:"<<std::setprecision(4)<<nll<<"\n";

    if((a>0) & AUTO_STOP){
      if(nll > nlls.back()){
        nlls.push_back(nll);
        if(VERBOSE) Rcpp::Rcout  <<"Stopped at attempt "<<a<<"\n";
        break;
      }
    }

    nlls.push_back(nll);
    stepsize0 *= SCALE;
  }

  clock.tock("main");
  clock.stop("clock");

  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("stepsizes") = stepsizes,
    Rcpp::Named("nlls") = nlls
  );

  return output;
}


//' @export
// [[Rcpp::export]]
Rcpp::List armbLR(
  const Eigen::Map<Eigen::VectorXd> Y,
  const Eigen::Map<Eigen::MatrixXd> X,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int R,
  const int MAXT,
  const int BURN,
  const int BATCH,
  const double STEPSIZE0,
  const double PAR1,
  const double PAR2,
  const double PAR3,
  const int STORE,
  const int SKIP_PRINT,
  const int SEED,
  const bool VERBOSE
){
  // Identify dimensions
  const int n = Y.size();
  const int m = THETA0.size();


  Eigen::MatrixXd r_theta(R, m);


  int last_iter = MAXT;
  for(int r = 0; r < R; r++){
    std::vector<int> r_idx = resampleN(n, r);
    Eigen::VectorXd theta = THETA0;
    Eigen::VectorXd av_theta = THETA0;
    int batch_start = 0;
    int shf_idx = 0;

    for(int t = 0; t < MAXT; t++){
      Rcpp::checkUserInterrupt();
      if((batch_start+BATCH)>n){
        Rcpp::Rcout << "Batch index problem. End idx:"<< batch_start+BATCH <<". Reshuffling the data.\n";
        last_iter = t;
        r_idx = shuffleIVec(r_idx, r+shf_idx);
        shf_idx++ ;
        batch_start = 0;
        // break;
      }
      const std::vector<int> idx_t = subsetIVec(r_idx, batch_start, BATCH);
      const Eigen::MatrixXd x_t = X(idx_t, Eigen::all);
      const Eigen::VectorXd y_t = Y(idx_t);
      const Eigen::VectorXd eta_t = x_t*theta;
      const Eigen::VectorXd ngr = logistic::ngr(y_t, x_t, eta_t, BATCH)/BATCH;

      double stepsize_t = STEPSIZE0 * PAR1 * pow(1 + PAR2*STEPSIZE0*t, -PAR3);
      theta -= stepsize_t * ngr;

      if(VERBOSE & (t % SKIP_PRINT == 0)) Rcpp::Rcout  <<"\rIter:"<<t<<"| Idxs:"<<batch_start<<"-"<<batch_start+BATCH <<"| norm:"<< std::setprecision(4) << ngr.norm() <<"\n";


      if(t <= BURN){
        if(t == (BURN-1)){
          r_idx = shuffleIVec(r_idx, r+shf_idx);
          shf_idx++ ;
          batch_start = 0;
          av_theta = theta;
        }else{
          av_theta = theta;
          batch_start += BATCH;
        }
      }else{
        av_theta = ( (t - BURN) * av_theta + theta ) / (t - BURN + 1);
        batch_start += BATCH;
      }
    }

    r_theta.row(r) = av_theta;
  }

  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("par") = r_theta
  );

  return output;
}



//' @export
// [[Rcpp::export]]
Rcpp::List armGLM(
  const Eigen::Map<Eigen::VectorXd> Y,
  const Eigen::Map<Eigen::MatrixXd> X,
  const std::string FAMILY,
  const std::string LINK,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int MAXT,
  const int BURN,
  const int BATCH,
  const double STEPSIZE0,
  const double PAR1,
  const double PAR2,
  const double PAR3,
  const int VERBOSE_WINDOW,
  const int PATH_WINDOW,
  const int SEED,
  const bool VERBOSE
){

  Response resp(FAMILY, LINK);

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("main");

  // Identify dimensions
  const int n = Y.size();
  const int m = THETA0.size();

  std::vector<int> path_iters;
  std::vector<Eigen::VectorXd> path_theta;
  std::vector<double> path_nll;

  Eigen::VectorXd theta = THETA0;
  Eigen::VectorXd avtheta = THETA0;
  int last_iter = 0;


  double nll = 10000;
  Eigen::MatrixXd x = X;
  Eigen::VectorXd y = Y;
  int idx = 0;
  int shf = 0;
  for(int t = 0; t <= MAXT; t++){
    Rcpp::checkUserInterrupt();

    // Current nll
    // double prev_nll = nll;
    // nll = resp.nll_(Y, resp.linkinv_(X*theta))/n;

    // Store previous iteration results
    if(((t)%PATH_WINDOW == 0) | (t==MAXT)){
      path_iters.push_back(t);
      path_nll.push_back(nll);
      path_theta.push_back(avtheta);
    }

    // Break at t == MAXT (t starts from 0)
    if(t==MAXT){
      last_iter = t;
      break;
    }

    if((idx+BATCH)>n){
      idx = 0;
      x = shuffleRows(X, SEED + shf);
      y = shuffleVec(Y, SEED + shf);
      shf++;
    }

    const Eigen::MatrixXd x_t = x(Eigen::seqN(idx, BATCH), Eigen::all);
    const Eigen::VectorXd y_t = y.segment(idx, BATCH);
    const Eigen::VectorXd ngr = x_t.transpose()*(resp.linkinv_(x_t*theta)-y_t)/BATCH;

    double stepsize_t = STEPSIZE0 * PAR1 * pow(1 + PAR2*STEPSIZE0*t, -PAR3);
    theta -= stepsize_t * ngr;

    if(t <= BURN){
      avtheta = theta;
    }else{
      avtheta = ( (t - BURN) * avtheta + theta ) / (t - BURN + 1);
    }

    idx += BATCH;
    last_iter++;
  }

  clock.tock("main");
  clock.stop("clock");

  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("path_theta") = path_theta,
    Rcpp::Named("path_iters") = path_iters,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("avtheta") = avtheta,
    Rcpp::Named("last_iter") = last_iter
    // Rcpp::Named("nll") = resp.nll_(Y, resp.linkinv_(X*theta))
  );

  return output;
}


//' @export
// [[Rcpp::export]]
Rcpp::List tune_armGLM(
  const Eigen::Map<Eigen::VectorXd> Y,
  const Eigen::Map<Eigen::MatrixXd> X,
  const std::string FAMILY,
  const std::string LINK,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int MAXT,
  const int BURN,
  const int BATCH,
  const double STEPSIZE0,
  const double SCALE,
  const double MAXA,
  const double PAR1,
  const double PAR2,
  const double PAR3,
  const bool AUTO_STOP,
  const int SKIP_PRINT,
  const int SEED,
  const bool VERBOSE
){

  Response resp(FAMILY, LINK);

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("main");

  // Identify dimensions
  const int n = Y.size();
  const int m = THETA0.size();
  const int maxt = std::min(double(MAXT), double(int(n / BATCH)));

  int last_update = 0;

  const Eigen::MatrixXd x = shuffleRows(X, SEED);
  std::vector<double> stepsizes;
  std::vector<double> devresids;

  double stepsize0 = STEPSIZE0;
  for(int a = 0; a < MAXA; a++){

    int idx = 0;
    if(VERBOSE) Rcpp::Rcout  <<"Stepsize:"<<std::setprecision(4)<<stepsize0<<" | ";

    Rcpp::List fit = armGLM(Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, stepsize0,
                           PAR1, PAR2, PAR3, 1, 10, SEED, false);

    stepsizes.push_back(stepsize0);
    Eigen::VectorXd avtheta = fit["avtheta"];
    double dev = resp.dev_(Y, resp.linkinv_(X*avtheta))/n;
    if(VERBOSE) Rcpp::Rcout  <<"dev:"<<std::setprecision(4)<<dev<<"\n";

    if((a>0) & AUTO_STOP){
      if(dev > devresids.back()){
        devresids.push_back(dev);
        if(VERBOSE) Rcpp::Rcout  <<"Stopped at attempt "<<a<<"\n";
        break;
      }
    }

    devresids.push_back(dev);
    stepsize0 *= SCALE;
  }

  clock.tock("main");
  clock.stop("clock");

  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("stepsizes") = stepsizes,
    Rcpp::Named("devresids") = devresids
  );

  return output;
}
