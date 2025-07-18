#ifndef logistic_ri_H
#define logistic_ri_H


// #include <Rcpp.h>
using namespace Rcpp;
// #include <cmath>
// #include <RcppEigen.h>
#include <random>

// [[Rcpp::export]]
double logsumexp(Rcpp::NumericVector x) {
  return(log(sum(exp(x - max(x)))) + max(x));
}



// [[Rcpp::export]]
double innerProduct(Rcpp::DoubleVector x, Rcpp::DoubleVector y) {
  return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}

// Individual marginal likelihood
// [[Rcpp::export]]
double likAGHi(Rcpp::DoubleVector beta, double logsigma,
              Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d,
              int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z, int i){
  int len = list_y.size();
  double sigma = exp(logsigma);
  double ll = 0.0;

  Rcpp::NumericMatrix xi = list_x[i];
  Rcpp::IntegerVector yi = list_y[i];
  Rcpp::IntegerVector di = list_d[i];
  int ni = yi.size();
  double H = 0.0;
  double ui = 0.0;
  for (int k=0; k<niter; k++){
    double g = -1.0 * ui;
    H = 1.0;
    for (int j=0; j<ni; j++){
      double arg = innerProduct(xi(j, _), beta) + sigma * ui;
      double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
      g += sigma * (yi[j] - di[j] * pij);
      H += pow(sigma, 2.0) * di[j] * pij * (1.0 - pij);
    }
    ui += g / H;
  }
  double se = 1.0 / sqrt(H);
  // now computes AGH
  int nq = ws.size();
  DoubleVector fi (nq);
  for (int k=0; k<nq; k++){
    double zk = sqrt(2.0) * se * z[k] + ui;
    for (int j=0; j<ni; j++){
      double arg = innerProduct(xi(j, _), beta) + sigma * zk;
      fi[k] += yi[j] * arg - di[j] * log(1.0 + exp(arg));
    }
    fi[k] += R::dnorm(zk, 0.0, 1.0, 1);
    fi[k] += log(ws[k]);
  }
  double logIi = logsumexp(fi);
  logIi += log(se);

  return(-logIi);
}

// Individual marginal likelihood gradient
// [[Rcpp::export]]
Eigen::VectorXd grAGHi(Eigen::VectorXd betavec, double logsigma,
                         Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d,
                         int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z, int i)
{

  SEXP b = Rcpp::wrap(betavec);
  Rcpp::DoubleVector beta(b);


  const unsigned int p = beta.size();
  int len = list_y.size();
  Rcpp::DoubleVector  out (p + 1);
  double sigma = exp(logsigma);
  // first locate the modes in u
  Rcpp::NumericMatrix xi = list_x[i];
  Rcpp::IntegerVector yi = list_y[i];
  Rcpp::IntegerVector di = list_d[i];
  int ni = yi.size();
  double H = 0.0;
  double ui = 0.0;
  for (int k=0; k<niter; k++){
    double g = -1.0 * ui;
    H = 1.0;
    for (int j=0; j<ni; j++){
      double arg = innerProduct(xi(j, _), beta) + sigma * ui;
      double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
      double vij = pij * (1.0 - pij);
      g += sigma * (yi[j] - di[j] * pij);
      H += pow(sigma, 2.0) * di[j] * vij;
    }
    ui += g / H;
  }
  // Rcpp::Rcout<<"ui:"<<ui<<"\n";
  Rcpp::DoubleVector dzibetanum (p);
  double dziden = 0.0;
  double dzisigmanum = 0.0;
  Rcpp::DoubleVector dh2beta (p);
  double dhzi = 0.0;
  double dh2sigma = 0.0;
  for (int j=0; j<ni; j++){
    double arg = innerProduct(xi(j, _), beta) + sigma * ui;
    double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
    double vij = pij * (1.0 - pij);
    dzibetanum += di[j] * sigma * vij * xi(j, _);
    dziden += di[j] * pow(sigma, 2.0) * vij;
    dzisigmanum += yi[j] - di[j] * pij - di[j] * vij * sigma * ui;
    dh2beta += di[j] * (1.0 - 2.0 * pij) * pow(sigma, 2.0) * vij * xi(j, _);
    dhzi += di[j] *  (1.0 - 2.0 * pij) *  pow(sigma, 3.0) * vij;
    dh2sigma += di[j] * (1.0 - 2.0 * pij) *  pow(sigma, 2.0) * vij * ui + di[j] * vij * 2.0 * sigma;
  }
  double se = 1.0 / sqrt(H);
  // Rcpp::Rcout<<"se:"<<se<<"\n";

  double dzisigma = dzisigmanum / (1.0 + dziden);
  // Rcpp::Rcout<<"dzisigma:"<<dzisigma<<"\n";

  Rcpp::DoubleVector dzibeta =  (-1.0) * dzibetanum / (1.0 + dziden);
  Rcpp::DoubleVector dsebeta = -0.5 * pow(H, -1.5) * (dh2beta + dhzi * dzibeta);
  double dsesigma =  -0.5 * pow(H, -1.5) * (dh2sigma + dhzi * dzisigma);
  // now computes grad AGH
  int nq = ws.size();
  DoubleVector fi (nq);
  Rcpp::DoubleVector numbetai (p);
  double numsigmai = 0.0;
  for (int k=0; k<nq; k++){
    double zk = sqrt(2.0) * se * z[k] + ui;
    DoubleVector dzkbeta =  sqrt(2.0) * dsebeta * z[k] + dzibeta;
    double dzksigma = sqrt(2.0) * dsesigma * z[k] + dzisigma;
    double dhzeta = 0.0;
    DoubleVector dhbeta (p);
    double dhsigma = 0.0;
    for (int j=0; j<ni; j++){
      double arg = innerProduct(xi(j, _), beta) + sigma * zk;
      double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
      fi[k] += yi[j] * arg - di[j] * log(1.0 + exp(arg));
      dhbeta += (yi[j] - di[j] * pij) * xi(j, _);
      dhsigma += (yi[j] - di[j] * pij) * zk;
      dhzeta += (yi[j] - di[j] * pij) * sigma;
    }
    dhzeta -= zk;
    fi[k] += R::dnorm(zk, 0.0, 1.0, 1);
    fi[k] += log(ws[k]);
    dhbeta += dhzeta * dzkbeta;
    dhsigma += dhzeta * dzksigma;
    numbetai += exp(fi[k]) * dhbeta;
    numsigmai += exp(fi[k]) * dhsigma;
  }
  double logIi = logsumexp(fi);
  // Rcpp::Rcout<<"logIi:"<<logIi<<"\n";
  // Rcpp::Rcout<<"numbetai:"<<numbetai<<"\n";
  // Rcpp::Rcout<<"dsebeta:"<<dsebeta<<"\n";

  // out[Range(0, p-1)] = dsebeta / se +  numbetai / exp(logIi);
  IntegerVector signsnumbetai=sign(numbetai);
  // Rcpp::Rcout<<"signsnumbetai:"<<signsnumbetai<<"\n";
  DoubleVector tmpbeta = exp(log(abs(numbetai)) -logIi);
  tmpbeta=tmpbeta*Rcpp::as<DoubleVector>(signsnumbetai);


  out[Range(0, p-1)] = dsebeta / se +  tmpbeta;

  // out[p] = dsesigma / se + numsigmai / exp(logIi);



  if(numsigmai>0){
    out[p] = dsesigma / se + exp(log(numsigmai)-logIi) ;
  }else{
    out[p] = dsesigma / se + -exp(log(-numsigmai)-logIi) ;
  }

  out[p] *= sigma;

  Eigen::Map<Eigen::VectorXd> outvec(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(out));

  return(-outvec);
}

// [[Rcpp::export]]
double likAGH(Eigen::VectorXd betavec, double sigma,
              Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d,
              int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z)
{

  SEXP b = Rcpp::wrap(betavec);
  Rcpp::DoubleVector beta(b);
  int len = list_y.size();
  double ll = 0.0;
  // first locate the modes in u
  for(int i=0; i<len; i++){
    Rcpp::NumericMatrix xi = list_x[i];
    Rcpp::IntegerVector yi = list_y[i];
    Rcpp::IntegerVector di = list_d[i];
    int ni = yi.size();
    double H = 0.0;
    double ui = 0.0;
    for (int k=0; k<niter; k++){
      double g = -1.0 * ui;
      H = 1.0;
      for (int j=0; j<ni; j++){
        double arg = innerProduct(xi(j, _), beta) + sigma * ui;
        double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
        g += sigma * (yi[j] - di[j] * pij);
        H += pow(sigma, 2.0) * di[j] * pij * (1.0 - pij);
       }
      ui += g / H;
    }
    double se = 1.0 / sqrt(H);
    // now computes AGH
    int nq = ws.size();
    DoubleVector fi (nq);
    for (int k=0; k<nq; k++){
      double zk = sqrt(2.0) * se * z[k] + ui;
      for (int j=0; j<ni; j++){
        double arg = innerProduct(xi(j, _), beta) + sigma * zk;
        fi[k] += yi[j] * arg - di[j] * log(1.0 + exp(arg));
        }
      fi[k] += R::dnorm(zk, 0.0, 1.0, 1);
      fi[k] += log(ws[k]);
    }
    double logIi = logsumexp(fi);
    logIi += log(se);
    ll += logIi;
  }
  return(ll);
}



// [[Rcpp::export]]
Rcpp::DoubleVector grAGH(Rcpp::DoubleVector beta, double sigma,
              Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d,
              int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z){
  const unsigned int p = beta.size();
  int len = list_y.size();
  Rcpp::DoubleVector  out (p + 1);
  // first locate the modes in u
  for(int i=0; i<len; i++){
    Rcpp::NumericMatrix xi = list_x[i];
    Rcpp::IntegerVector yi = list_y[i];
    Rcpp::IntegerVector di = list_d[i];
    int ni = yi.size();
    double H = 0.0;
    double ui = 0.0;
    for (int k=0; k<niter; k++){
      double g = -1.0 * ui;
      H = 1.0;
      for (int j=0; j<ni; j++){
        double arg = innerProduct(xi(j, _), beta) + sigma * ui;
        double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
        double vij = pij * (1.0 - pij);
        g += sigma * (yi[j] - di[j] * pij);
        H += pow(sigma, 2.0) * di[j] * vij;
      }
      ui += g / H;
    }
    Rcpp::DoubleVector dzibetanum (p);
    double dziden = 0.0;
    double dzisigmanum = 0.0;
    Rcpp::DoubleVector dh2beta (p);
    double dhzi = 0.0;
    double dh2sigma = 0.0;
    for (int j=0; j<ni; j++){
      double arg = innerProduct(xi(j, _), beta) + sigma * ui;
      double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
      double vij = pij * (1.0 - pij);
      dzibetanum += di[j] * sigma * vij * xi(j, _);
      dziden += di[j] * pow(sigma, 2.0) * vij;
      dzisigmanum += yi[j] - di[j] * pij - di[j] * vij * sigma * ui;
      dh2beta += di[j] * (1.0 - 2.0 * pij) * pow(sigma, 2.0) * vij * xi(j, _);
      dhzi += di[j] *  (1.0 - 2.0 * pij) *  pow(sigma, 3.0) * vij;
      dh2sigma += di[j] * (1.0 - 2.0 * pij) *  pow(sigma, 2.0) * vij * ui + di[j] * vij * 2.0 * sigma;
    }
    double se = 1.0 / sqrt(H);
    double dzisigma = dzisigmanum / (1.0 + dziden);
    Rcpp::DoubleVector dzibeta =  (-1.0) * dzibetanum / (1.0 + dziden);
    Rcpp::DoubleVector dsebeta = -0.5 * pow(H, -1.5) * (dh2beta + dhzi * dzibeta);
    double dsesigma =  -0.5 * pow(H, -1.5) * (dh2sigma + dhzi * dzisigma);
   // now computes grad AGH
    int nq = ws.size();
    DoubleVector fi (nq);
    Rcpp::DoubleVector numbetai (p);
    double numsigmai = 0.0;
    for (int k=0; k<nq; k++){
      double zk = sqrt(2.0) * se * z[k] + ui;
      DoubleVector dzkbeta =  sqrt(2.0) * dsebeta * z[k] + dzibeta;
      double dzksigma = sqrt(2.0) * dsesigma * z[k] + dzisigma;
      double dhzeta = 0.0;
      DoubleVector dhbeta (p);
      double dhsigma = 0.0;
      for (int j=0; j<ni; j++){
        double arg = innerProduct(xi(j, _), beta) + sigma * zk;
        double pij = R::plogis(arg, 0.0, 1.0, 1, 0);
        fi[k] += yi[j] * arg - di[j] * log(1.0 + exp(arg));
        dhbeta += (yi[j] - di[j] * pij) * xi(j, _);
        dhsigma += (yi[j] - di[j] * pij) * zk;
        dhzeta += (yi[j] - di[j] * pij) * sigma;
        }
      dhzeta -= zk;
      fi[k] += R::dnorm(zk, 0.0, 1.0, 1);
      fi[k] += log(ws[k]);
      dhbeta += dhzeta * dzkbeta;
      dhsigma += dhzeta * dzksigma;
      numbetai += exp(fi[k]) * dhbeta;
      numsigmai += exp(fi[k]) * dhsigma;
    }
    double logIi = logsumexp(fi);
    out[Range(0, p-1)] += dsebeta / se +  numbetai / exp(logIi);
    out[p] += dsesigma / se + numsigmai / exp(logIi);
  }
  return(out);
}


//' @export
// [[Rcpp::export]]
Rcpp::List armLOGRI(
  Rcpp::List LIST_X,
  Rcpp::List LIST_Y,
  Rcpp::List LIST_D,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int AGH_NITER,
  Rcpp::DoubleVector WS,
  Rcpp::DoubleVector Z,
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
  const bool VERBOSE,
  const int CONV_WINDOW = 1000,
  const bool CONV_CHECK = false,
  const double TOL = 1e-5
){




  // Identify dimensions
  int n = LIST_Y.size();
  const int m = THETA0.size();
  const int p = m-1;

  std::vector<int> path_iters;
  std::vector<Eigen::VectorXd> path_theta;
  std::vector<double> path_nll;
  std::vector<int> path_obs;

  Eigen::VectorXd theta = THETA0;
  Eigen::VectorXd avtheta = THETA0;
  int last_iter = 0;


  double nll = 10000;
  double prev_nll;
  bool convergence = false;
  int idx = 0;
  int shf = 0;
  std::vector<int> pool(n) ;
  std::iota(std::begin(pool), std::end(pool), 0);
  std::mt19937 randomizer(SEED);
  std::shuffle(pool.begin(), pool.end(), randomizer);
  for(int t = 0; t <= MAXT; t++){
    Rcpp::checkUserInterrupt();

    // Store previous iteration results
    if(((t)%PATH_WINDOW == 0) | (t==MAXT)){
      path_iters.push_back(t);
      path_nll.push_back(nll);
      path_theta.push_back(avtheta);
    }

    if(CONV_CHECK & (t>BURN)){
      if(t%CONV_WINDOW == 0){
        // Current nll
        prev_nll = nll;
        // nll = resp.nll_(Y, resp.linkinv_(X*avtheta))/n;
        // if(VERBOSE) Rcpp::Rcout <<"Iter "<< t<< ", nll: " << nll << "\n";
        //
        // if((prev_nll-nll)/CONV_WINDOW <=TOL | (nll-prev_nll)>0){
        //   convergence = true;
        //   last_iter = t;
        //   break;
        // }
      }
    }
    // Break at t == MAXT (t starts from 0)

    if(t==MAXT){
      last_iter = t;
      break;
    }

    if((idx+BATCH)>n){
      idx = 0;
      // x = shuffleRows(X, SEED + shf);
      // y = shuffleVec(Y, SEED + shf);
      std::mt19937 randomizer(SEED+shf);
      std::shuffle(pool.begin(), pool.end(), randomizer);
      shf++;
    }

    int obs = pool.at(idx);
    path_obs.push_back(obs);

    Eigen::VectorXd ngr = grAGHi(theta.segment(0,p), theta(p), LIST_X, LIST_Y, LIST_D,
           AGH_NITER, WS, Z, obs);

    // ngr.tail(1)/=10;
    double stepsize_t = STEPSIZE0 * PAR1 * pow(1 + PAR2*STEPSIZE0*(t+1), -PAR3);
    theta -= stepsize_t * ngr;

    if(t <= BURN){
      avtheta = theta;
    }else{
      avtheta = ( (t - BURN) * avtheta + theta ) / (t - BURN + 1);
    }

    idx += BATCH;
    last_iter++;
  }


  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("path_theta") = path_theta,
    Rcpp::Named("path_iters") = path_iters,
    Rcpp::Named("path_nll") = path_nll,
    Rcpp::Named("path_obs") = path_obs,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("avtheta") = avtheta,
    Rcpp::Named("last_iter") = last_iter,
    Rcpp::Named("convergence") = convergence
  );

  return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List tune_armLOGRI(
  Rcpp::List LIST_X,
  Rcpp::List LIST_Y,
  Rcpp::List LIST_D,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int AGH_NITER,
  Rcpp::DoubleVector WS,
  Rcpp::DoubleVector Z,
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




  // Identify dimensions
  int n = LIST_Y.size();
  const int m = THETA0.size();
  const int p = m-1;

  std::vector<int> path_iters;
  std::vector<Eigen::VectorXd> path_theta;
  std::vector<double> path_nll;
  std::vector<int> path_obs;

  Eigen::VectorXd theta = THETA0;
  Eigen::VectorXd avtheta = THETA0;
  int last_iter = 0;


  double nll = 10000;
  double prev_nll;
  bool convergence = false;
  int idx = 0;
  int shf = 0;
  std::vector<int> pool(n) ;
  std::iota(std::begin(pool), std::end(pool), 0);
  std::mt19937 randomizer(SEED);
  std::shuffle(pool.begin(), pool.end(), randomizer);

  std::vector<double> stepsizes;
  std::vector<double> devresids;

  double stepsize0 = STEPSIZE0;
  for(int a = 0; a < MAXA; a++){

    int idx = 0;
    if(VERBOSE) Rcpp::Rcout  <<"Stepsize:"<<std::setprecision(4)<<stepsize0<<" | ";

    Rcpp::List fit = armLOGRI(LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z,
                              MAXT, BURN, BATCH, stepsize0, PAR1, PAR2, PAR3,
                              1, 10, SEED, false, 1000, false, 1e-5);

    stepsizes.push_back(stepsize0);
    Eigen::VectorXd avtheta = fit["avtheta"];
    double llik = -likAGH(avtheta.segment(0,p), avtheta(p), LIST_X, LIST_Y, LIST_D,
                         AGH_NITER, WS, Z)/n;
    if(VERBOSE) Rcpp::Rcout  <<"llik:"<<std::setprecision(4)<<llik<<"\n";

    if((a>0) & AUTO_STOP){
      if(llik > devresids.back()){
        devresids.push_back(llik);
        if(VERBOSE) Rcpp::Rcout  <<"Stopped at attempt "<<a+1<<"\n";
        break;
      }
    }
    devresids.push_back(llik);
    stepsize0 *= SCALE;
  }


  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("stepsizes") = stepsizes,
    Rcpp::Named("devresids") = devresids
  );

  return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List armLOGRI2(
  Rcpp::List LIST_X,
  Rcpp::List LIST_Y,
  Rcpp::List LIST_D,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int AGH_NITER,
  Rcpp::DoubleVector WS,
  Rcpp::DoubleVector Z,
  const int LENGTH,
  const int BURN,
  const double STEPSIZE0,
  const int SEED,
  const bool VERBOSE,
  const int TRIM=100,
  const int CONV_WINDOW = 3,
  const bool CONV_CHECK = true,
  const double TOL = 1e-2
){




  // Identify dimensions
  int n = LIST_Y.size();
  const int m = THETA0.size();
  const int p = m-1;

  Eigen::VectorXd mle = THETA0;
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(m);
  Eigen::VectorXd avdelta = delta;
  Eigen::VectorXd prev_delta;

  int last_iter = 0;
  int burn = BURN;
  double nll_start = 1000;
  double norm_delta = 0;
  double max_diff = 0;
  double max_ngr = 0;
  double max_pdiff = 0;
  double nll = nll_start;
  double prev_nll = nll_start;
  double pf=1;

  std::vector<int> path_iters; path_iters.push_back(0);
  std::vector<Eigen::VectorXd> path_theta; path_theta.push_back(mle);
  std::vector<Eigen::VectorXd> path_delta; path_delta.push_back(delta);
  std::vector<double> path_nll; path_nll.push_back(nll_start);
  std::vector<double> path_pf; path_pf.push_back(pf);
  std::vector<double> path_norm; path_norm.push_back(norm_delta);
  std::vector<double> path_diff; path_diff.push_back(max_diff);
  std::vector<double> path_pdiff; path_pdiff.push_back(max_pdiff);
  std::vector<double> path_ngr; path_ngr.push_back(max_ngr);

  bool convergence = false;
  int idx = 0;
  int shf = 0;
  std::vector<int> pool(n) ;
  std::iota(std::begin(pool), std::end(pool), 0);
  std::mt19937 randomizer(SEED);
  std::shuffle(pool.begin(), pool.end(), randomizer);
  int conv_counter =0;
  for(int t = 1; t < (burn + LENGTH); t++){
    Rcpp::checkUserInterrupt();

    if((idx+1)>n){
      idx = 0;
      std::mt19937 randomizer(SEED+shf);
      std::shuffle(pool.begin(), pool.end(), randomizer);
      shf++;
    }

    int obs = pool.at(idx);
    // path_obs.push_back(obs);

    Eigen::VectorXd theta = delta+mle;
    // Rcpp::Rcout << theta << "\n";
    // Rcpp::Rcout << "m:" << m<<", p:"<<p << ", n:"<<n<< "\n";
    Eigen::VectorXd ngr = grAGHi(theta.segment(0,p), theta(p), LIST_X, LIST_Y, LIST_D,
           AGH_NITER, WS, Z, obs);
           
    double stepsize_t = STEPSIZE0 * pow(t, -.5001);
    delta -= stepsize_t * ngr;

    if(t%TRIM == 0){

      nll = -likAGH(theta.segment(0,p), exp(theta(p)), LIST_X, LIST_Y, LIST_D,
           AGH_NITER, WS, Z);

      if(1){
        pf = abs((nll-prev_nll)/prev_nll);
        norm_delta = delta.squaredNorm()/m;
        max_ngr = ngr.cwiseAbs().maxCoeff();
        max_diff = max_ngr * stepsize_t;
        
        if(t < burn & CONV_CHECK & pf <= (TOL) ){
          conv_counter++;
          if(conv_counter==CONV_WINDOW){
            convergence = true;
            burn = t;
            idx = 0;
            std::mt19937 randomizer(SEED+shf);
            std::shuffle(pool.begin(), pool.end(), randomizer);
            shf++;
          }
        }else{
          conv_counter=0;
        }

        prev_nll=nll;
      }
      prev_delta=delta;
    }

    if(t <= burn){
      avdelta = delta;
    }else{
      avdelta = ( (t - burn-1) * avdelta + delta ) / (t - burn);
    }

    if(t%TRIM == 0){
      if(VERBOSE) Rcpp::Rcout << "Iter " << t << " | Dt L2: " << norm_delta << " | Dt diff LInf: " << max_diff <<"\n";
      path_iters.push_back(t);
      path_nll.push_back(nll);
      // path_theta.push_back(delta+mle);
      path_delta.push_back(avdelta);
      path_norm.push_back(norm_delta);
      path_diff.push_back(max_diff);
      path_ngr.push_back(max_ngr);
      path_pf.push_back(pf);
      // path_pdiff.push_back(max_pdiff);

    }

    idx ++;
    last_iter++;
  }

  Eigen::VectorXd avtheta = avdelta+mle;
  double nll_end = -likAGH(avtheta.segment(0,p), exp(avtheta(p)), LIST_X, LIST_Y, LIST_D,
            AGH_NITER, WS, Z);




    


    Rcpp::List output = Rcpp::List::create(
    // Rcpp::Named("path_theta") = path_theta,
    Rcpp::Named("path_delta") = path_delta,
    Rcpp::Named("path_iters") = path_iters,
    Rcpp::Named("nll_start") = nll_start,
    Rcpp::Named("nll_end")   = nll_end,
    Rcpp::Named("path_norm") = path_norm,
    Rcpp::Named("path_diff") = path_diff,
    Rcpp::Named("path_pdiff") = path_pdiff,
    Rcpp::Named("path_nll") = path_nll,
    Rcpp::Named("path_pf") = path_pf,
    Rcpp::Named("path_ngr") = path_ngr,
    Rcpp::Named("delta") = delta,
    Rcpp::Named("avdelta") = avdelta,
    Rcpp::Named("burn") = burn,
    Rcpp::Named("last_iter") = last_iter,
    Rcpp::Named("convergence") = convergence,
    Rcpp::Named("shf") = shf
  );

  return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List tune_armLOGRI2(
  Rcpp::List LIST_X,
  Rcpp::List LIST_Y,
  Rcpp::List LIST_D,
  const Eigen::Map<Eigen::VectorXd> THETA0,
  const int AGH_NITER,
  Rcpp::DoubleVector WS,
  Rcpp::DoubleVector Z,
  const int LENGTH,
  const int BURN,
  const double STEPSIZE0,
  const double SCALE,
  const double MAXA,
  const bool AUTO_STOP,
  const int SEED,
  const bool VERBOSE,
  const int TRIM=100,
  const int CONV_WINDOW = 10,
  const bool CONV_CHECK = true,
  const double TOL = 1e-2
){




  // Identify dimensions
  int n = LIST_Y.size();
  const int m = THETA0.size();
  const int p = m-1;

  std::vector<double> stepsizes;
  std::vector<double> nlls;
  double stepsize0 = STEPSIZE0;
  for(int a = 0; a < MAXA; a++){

    if(VERBOSE) Rcpp::Rcout  <<"Stepsize:"<<std::setprecision(4)<<stepsize0<<" | ";

    Rcpp::List fit = armLOGRI2(LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z,
                              LENGTH, BURN, stepsize0,
                              SEED, false, n, CONV_WINDOW, CONV_CHECK, TOL);

    stepsizes.push_back(stepsize0);
    double nll = fit["nll_end"];
    if(VERBOSE) Rcpp::Rcout  <<"llik:"<<std::setprecision(4)<<nll<<"\n";

    if((a>0) & AUTO_STOP){
      if(nll > nlls.back()){
        nlls.push_back(nll);
        if(VERBOSE) Rcpp::Rcout  <<"Stopped at attempt "<<a+1<<"\n";
        break;
      }
    }
    nlls.push_back(nll);
    stepsize0 *= SCALE;
  }


  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("stepsizes") = stepsizes,
    Rcpp::Named("nlls") = nlls
  );

  return output;
}
#endif
