// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// logistic_wrapper
Rcpp::List logistic_wrapper(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> THETA);
RcppExport SEXP _armb_logistic_wrapper(SEXP YSEXP, SEXP XSEXP, SEXP THETASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA(THETASEXP);
    rcpp_result_gen = Rcpp::wrap(logistic_wrapper(Y, X, THETA));
    return rcpp_result_gen;
END_RCPP
}
// logsumexp
double logsumexp(Rcpp::NumericVector x);
RcppExport SEXP _armb_logsumexp(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(logsumexp(x));
    return rcpp_result_gen;
END_RCPP
}
// innerProduct
double innerProduct(Rcpp::DoubleVector x, Rcpp::DoubleVector y);
RcppExport SEXP _armb_innerProduct(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(innerProduct(x, y));
    return rcpp_result_gen;
END_RCPP
}
// likAGHi
double likAGHi(Rcpp::DoubleVector beta, double logsigma, Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d, int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z, int i);
RcppExport SEXP _armb_likAGHi(SEXP betaSEXP, SEXP logsigmaSEXP, SEXP list_xSEXP, SEXP list_ySEXP, SEXP list_dSEXP, SEXP niterSEXP, SEXP wsSEXP, SEXP zSEXP, SEXP iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type logsigma(logsigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_x(list_xSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_y(list_ySEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_d(list_dSEXP);
    Rcpp::traits::input_parameter< int >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    rcpp_result_gen = Rcpp::wrap(likAGHi(beta, logsigma, list_x, list_y, list_d, niter, ws, z, i));
    return rcpp_result_gen;
END_RCPP
}
// grAGHi
Eigen::VectorXd grAGHi(Eigen::VectorXd betavec, double logsigma, Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d, int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z, int i);
RcppExport SEXP _armb_grAGHi(SEXP betavecSEXP, SEXP logsigmaSEXP, SEXP list_xSEXP, SEXP list_ySEXP, SEXP list_dSEXP, SEXP niterSEXP, SEXP wsSEXP, SEXP zSEXP, SEXP iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type betavec(betavecSEXP);
    Rcpp::traits::input_parameter< double >::type logsigma(logsigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_x(list_xSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_y(list_ySEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_d(list_dSEXP);
    Rcpp::traits::input_parameter< int >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    rcpp_result_gen = Rcpp::wrap(grAGHi(betavec, logsigma, list_x, list_y, list_d, niter, ws, z, i));
    return rcpp_result_gen;
END_RCPP
}
// likAGH
double likAGH(Eigen::VectorXd betavec, double sigma, Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d, int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z);
RcppExport SEXP _armb_likAGH(SEXP betavecSEXP, SEXP sigmaSEXP, SEXP list_xSEXP, SEXP list_ySEXP, SEXP list_dSEXP, SEXP niterSEXP, SEXP wsSEXP, SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type betavec(betavecSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_x(list_xSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_y(list_ySEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_d(list_dSEXP);
    Rcpp::traits::input_parameter< int >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(likAGH(betavec, sigma, list_x, list_y, list_d, niter, ws, z));
    return rcpp_result_gen;
END_RCPP
}
// grAGH
Rcpp::DoubleVector grAGH(Rcpp::DoubleVector beta, double sigma, Rcpp::List list_x, Rcpp::List list_y, Rcpp::List list_d, int niter, Rcpp::DoubleVector ws, Rcpp::DoubleVector z);
RcppExport SEXP _armb_grAGH(SEXP betaSEXP, SEXP sigmaSEXP, SEXP list_xSEXP, SEXP list_ySEXP, SEXP list_dSEXP, SEXP niterSEXP, SEXP wsSEXP, SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_x(list_xSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_y(list_ySEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type list_d(list_dSEXP);
    Rcpp::traits::input_parameter< int >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(grAGH(beta, sigma, list_x, list_y, list_d, niter, ws, z));
    return rcpp_result_gen;
END_RCPP
}
// armLOGRI
Rcpp::List armLOGRI(Rcpp::List LIST_X, Rcpp::List LIST_Y, Rcpp::List LIST_D, const Eigen::Map<Eigen::VectorXd> THETA0, const int AGH_NITER, Rcpp::DoubleVector WS, Rcpp::DoubleVector Z, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double PAR1, const double PAR2, const double PAR3, const int VERBOSE_WINDOW, const int PATH_WINDOW, const int SEED, const bool VERBOSE, const int CONV_WINDOW, const bool CONV_CHECK, const double TOL);
RcppExport SEXP _armb_armLOGRI(SEXP LIST_XSEXP, SEXP LIST_YSEXP, SEXP LIST_DSEXP, SEXP THETA0SEXP, SEXP AGH_NITERSEXP, SEXP WSSEXP, SEXP ZSEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP VERBOSE_WINDOWSEXP, SEXP PATH_WINDOWSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP, SEXP CONV_WINDOWSEXP, SEXP CONV_CHECKSEXP, SEXP TOLSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type LIST_X(LIST_XSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type LIST_Y(LIST_YSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type LIST_D(LIST_DSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type AGH_NITER(AGH_NITERSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type WS(WSSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const int >::type VERBOSE_WINDOW(VERBOSE_WINDOWSEXP);
    Rcpp::traits::input_parameter< const int >::type PATH_WINDOW(PATH_WINDOWSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    Rcpp::traits::input_parameter< const int >::type CONV_WINDOW(CONV_WINDOWSEXP);
    Rcpp::traits::input_parameter< const bool >::type CONV_CHECK(CONV_CHECKSEXP);
    Rcpp::traits::input_parameter< const double >::type TOL(TOLSEXP);
    rcpp_result_gen = Rcpp::wrap(armLOGRI(LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, VERBOSE_WINDOW, PATH_WINDOW, SEED, VERBOSE, CONV_WINDOW, CONV_CHECK, TOL));
    return rcpp_result_gen;
END_RCPP
}
// tune_armLOGRI
Rcpp::List tune_armLOGRI(Rcpp::List LIST_X, Rcpp::List LIST_Y, Rcpp::List LIST_D, const Eigen::Map<Eigen::VectorXd> THETA0, const int AGH_NITER, Rcpp::DoubleVector WS, Rcpp::DoubleVector Z, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double SCALE, const double MAXA, const double PAR1, const double PAR2, const double PAR3, const bool AUTO_STOP, const int SKIP_PRINT, const int SEED, const bool VERBOSE);
RcppExport SEXP _armb_tune_armLOGRI(SEXP LIST_XSEXP, SEXP LIST_YSEXP, SEXP LIST_DSEXP, SEXP THETA0SEXP, SEXP AGH_NITERSEXP, SEXP WSSEXP, SEXP ZSEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP SCALESEXP, SEXP MAXASEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP AUTO_STOPSEXP, SEXP SKIP_PRINTSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type LIST_X(LIST_XSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type LIST_Y(LIST_YSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type LIST_D(LIST_DSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type AGH_NITER(AGH_NITERSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type WS(WSSEXP);
    Rcpp::traits::input_parameter< Rcpp::DoubleVector >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type SCALE(SCALESEXP);
    Rcpp::traits::input_parameter< const double >::type MAXA(MAXASEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const bool >::type AUTO_STOP(AUTO_STOPSEXP);
    Rcpp::traits::input_parameter< const int >::type SKIP_PRINT(SKIP_PRINTSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    rcpp_result_gen = Rcpp::wrap(tune_armLOGRI(LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE));
    return rcpp_result_gen;
END_RCPP
}
// armLR2
Rcpp::List armLR2(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> THETA0, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double PAR1, const double PAR2, const double PAR3, const int STORE, const int SKIP_PRINT, const int SEED, const bool VERBOSE);
RcppExport SEXP _armb_armLR2(SEXP YSEXP, SEXP XSEXP, SEXP THETA0SEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP STORESEXP, SEXP SKIP_PRINTSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const int >::type STORE(STORESEXP);
    Rcpp::traits::input_parameter< const int >::type SKIP_PRINT(SKIP_PRINTSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    rcpp_result_gen = Rcpp::wrap(armLR2(Y, X, THETA0, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, STORE, SKIP_PRINT, SEED, VERBOSE));
    return rcpp_result_gen;
END_RCPP
}
// tune_armLR
Rcpp::List tune_armLR(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> THETA0, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double SCALE, const double MAXA, const double PAR1, const double PAR2, const double PAR3, const bool AUTO_STOP, const int SKIP_PRINT, const int SEED, const bool VERBOSE);
RcppExport SEXP _armb_tune_armLR(SEXP YSEXP, SEXP XSEXP, SEXP THETA0SEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP SCALESEXP, SEXP MAXASEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP AUTO_STOPSEXP, SEXP SKIP_PRINTSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type SCALE(SCALESEXP);
    Rcpp::traits::input_parameter< const double >::type MAXA(MAXASEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const bool >::type AUTO_STOP(AUTO_STOPSEXP);
    Rcpp::traits::input_parameter< const int >::type SKIP_PRINT(SKIP_PRINTSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    rcpp_result_gen = Rcpp::wrap(tune_armLR(Y, X, THETA0, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE));
    return rcpp_result_gen;
END_RCPP
}
// armbLR
Rcpp::List armbLR(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> THETA0, const int R, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double PAR1, const double PAR2, const double PAR3, const int STORE, const int SKIP_PRINT, const int SEED, const bool VERBOSE);
RcppExport SEXP _armb_armbLR(SEXP YSEXP, SEXP XSEXP, SEXP THETA0SEXP, SEXP RSEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP STORESEXP, SEXP SKIP_PRINTSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type R(RSEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const int >::type STORE(STORESEXP);
    Rcpp::traits::input_parameter< const int >::type SKIP_PRINT(SKIP_PRINTSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    rcpp_result_gen = Rcpp::wrap(armbLR(Y, X, THETA0, R, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, STORE, SKIP_PRINT, SEED, VERBOSE));
    return rcpp_result_gen;
END_RCPP
}
// armGLM
Rcpp::List armGLM(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, const std::string FAMILY, const std::string LINK, const Eigen::Map<Eigen::VectorXd> THETA0, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double PAR1, const double PAR2, const double PAR3, const int VERBOSE_WINDOW, const int PATH_WINDOW, const int SEED, const bool VERBOSE, const int CONV_WINDOW, const bool CONV_CHECK, const double TOL);
RcppExport SEXP _armb_armGLM(SEXP YSEXP, SEXP XSEXP, SEXP FAMILYSEXP, SEXP LINKSEXP, SEXP THETA0SEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP VERBOSE_WINDOWSEXP, SEXP PATH_WINDOWSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP, SEXP CONV_WINDOWSEXP, SEXP CONV_CHECKSEXP, SEXP TOLSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::string >::type FAMILY(FAMILYSEXP);
    Rcpp::traits::input_parameter< const std::string >::type LINK(LINKSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const int >::type VERBOSE_WINDOW(VERBOSE_WINDOWSEXP);
    Rcpp::traits::input_parameter< const int >::type PATH_WINDOW(PATH_WINDOWSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    Rcpp::traits::input_parameter< const int >::type CONV_WINDOW(CONV_WINDOWSEXP);
    Rcpp::traits::input_parameter< const bool >::type CONV_CHECK(CONV_CHECKSEXP);
    Rcpp::traits::input_parameter< const double >::type TOL(TOLSEXP);
    rcpp_result_gen = Rcpp::wrap(armGLM(Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, VERBOSE_WINDOW, PATH_WINDOW, SEED, VERBOSE, CONV_WINDOW, CONV_CHECK, TOL));
    return rcpp_result_gen;
END_RCPP
}
// tune_armGLM
Rcpp::List tune_armGLM(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, const std::string FAMILY, const std::string LINK, const Eigen::Map<Eigen::VectorXd> THETA0, const int MAXT, const int BURN, const int BATCH, const double STEPSIZE0, const double SCALE, const double MAXA, const double PAR1, const double PAR2, const double PAR3, const bool AUTO_STOP, const int SKIP_PRINT, const int SEED, const bool VERBOSE);
RcppExport SEXP _armb_tune_armGLM(SEXP YSEXP, SEXP XSEXP, SEXP FAMILYSEXP, SEXP LINKSEXP, SEXP THETA0SEXP, SEXP MAXTSEXP, SEXP BURNSEXP, SEXP BATCHSEXP, SEXP STEPSIZE0SEXP, SEXP SCALESEXP, SEXP MAXASEXP, SEXP PAR1SEXP, SEXP PAR2SEXP, SEXP PAR3SEXP, SEXP AUTO_STOPSEXP, SEXP SKIP_PRINTSEXP, SEXP SEEDSEXP, SEXP VERBOSESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::string >::type FAMILY(FAMILYSEXP);
    Rcpp::traits::input_parameter< const std::string >::type LINK(LINKSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA0(THETA0SEXP);
    Rcpp::traits::input_parameter< const int >::type MAXT(MAXTSEXP);
    Rcpp::traits::input_parameter< const int >::type BURN(BURNSEXP);
    Rcpp::traits::input_parameter< const int >::type BATCH(BATCHSEXP);
    Rcpp::traits::input_parameter< const double >::type STEPSIZE0(STEPSIZE0SEXP);
    Rcpp::traits::input_parameter< const double >::type SCALE(SCALESEXP);
    Rcpp::traits::input_parameter< const double >::type MAXA(MAXASEXP);
    Rcpp::traits::input_parameter< const double >::type PAR1(PAR1SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR2(PAR2SEXP);
    Rcpp::traits::input_parameter< const double >::type PAR3(PAR3SEXP);
    Rcpp::traits::input_parameter< const bool >::type AUTO_STOP(AUTO_STOPSEXP);
    Rcpp::traits::input_parameter< const int >::type SKIP_PRINT(SKIP_PRINTSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    rcpp_result_gen = Rcpp::wrap(tune_armGLM(Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE));
    return rcpp_result_gen;
END_RCPP
}
// test_glm
Rcpp::List test_glm(const Eigen::Map<Eigen::VectorXd> Y, const Eigen::Map<Eigen::MatrixXd> X, std::string FAMILY, std::string LINK, const Eigen::Map<Eigen::VectorXd> THETA);
RcppExport SEXP _armb_test_glm(SEXP YSEXP, SEXP XSEXP, SEXP FAMILYSEXP, SEXP LINKSEXP, SEXP THETASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::string >::type FAMILY(FAMILYSEXP);
    Rcpp::traits::input_parameter< std::string >::type LINK(LINKSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type THETA(THETASEXP);
    rcpp_result_gen = Rcpp::wrap(test_glm(Y, X, FAMILY, LINK, THETA));
    return rcpp_result_gen;
END_RCPP
}
// shuffleRows
Eigen::MatrixXd shuffleRows(const Eigen::MatrixXd& X, const int SEED);
RcppExport SEXP _armb_shuffleRows(SEXP XSEXP, SEXP SEEDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    rcpp_result_gen = Rcpp::wrap(shuffleRows(X, SEED));
    return rcpp_result_gen;
END_RCPP
}
// shuffleVec
Eigen::VectorXd shuffleVec(const Eigen::VectorXd& X, const int SEED);
RcppExport SEXP _armb_shuffleVec(SEXP XSEXP, SEXP SEEDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    rcpp_result_gen = Rcpp::wrap(shuffleVec(X, SEED));
    return rcpp_result_gen;
END_RCPP
}
// resampleN
std::vector<int> resampleN(const int N, const int SEED);
RcppExport SEXP _armb_resampleN(SEXP NSEXP, SEXP SEEDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    rcpp_result_gen = Rcpp::wrap(resampleN(N, SEED));
    return rcpp_result_gen;
END_RCPP
}
// sliceVec
Eigen::VectorXd sliceVec(const std::vector<int> SLICE_IDX, const Eigen::VectorXd X);
RcppExport SEXP _armb_sliceVec(SEXP SLICE_IDXSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int> >::type SLICE_IDX(SLICE_IDXSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(sliceVec(SLICE_IDX, X));
    return rcpp_result_gen;
END_RCPP
}
// sliceMat
Eigen::MatrixXd sliceMat(const std::vector<int> SLICE_IDX, const Eigen::MatrixXd X);
RcppExport SEXP _armb_sliceMat(SEXP SLICE_IDXSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int> >::type SLICE_IDX(SLICE_IDXSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(sliceMat(SLICE_IDX, X));
    return rcpp_result_gen;
END_RCPP
}
// subsetIVec
std::vector<int> subsetIVec(const std::vector<int>& X, const int START, const int LEN);
RcppExport SEXP _armb_subsetIVec(SEXP XSEXP, SEXP STARTSEXP, SEXP LENSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type START(STARTSEXP);
    Rcpp::traits::input_parameter< const int >::type LEN(LENSEXP);
    rcpp_result_gen = Rcpp::wrap(subsetIVec(X, START, LEN));
    return rcpp_result_gen;
END_RCPP
}
// shuffleIVec
std::vector<int> shuffleIVec(std::vector<int>& X, const int SEED);
RcppExport SEXP _armb_shuffleIVec(SEXP XSEXP, SEXP SEEDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<int>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type SEED(SEEDSEXP);
    rcpp_result_gen = Rcpp::wrap(shuffleIVec(X, SEED));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_armb_logistic_wrapper", (DL_FUNC) &_armb_logistic_wrapper, 3},
    {"_armb_logsumexp", (DL_FUNC) &_armb_logsumexp, 1},
    {"_armb_innerProduct", (DL_FUNC) &_armb_innerProduct, 2},
    {"_armb_likAGHi", (DL_FUNC) &_armb_likAGHi, 9},
    {"_armb_grAGHi", (DL_FUNC) &_armb_grAGHi, 9},
    {"_armb_likAGH", (DL_FUNC) &_armb_likAGH, 8},
    {"_armb_grAGH", (DL_FUNC) &_armb_grAGH, 8},
    {"_armb_armLOGRI", (DL_FUNC) &_armb_armLOGRI, 21},
    {"_armb_tune_armLOGRI", (DL_FUNC) &_armb_tune_armLOGRI, 20},
    {"_armb_armLR2", (DL_FUNC) &_armb_armLR2, 14},
    {"_armb_tune_armLR", (DL_FUNC) &_armb_tune_armLR, 16},
    {"_armb_armbLR", (DL_FUNC) &_armb_armbLR, 15},
    {"_armb_armGLM", (DL_FUNC) &_armb_armGLM, 19},
    {"_armb_tune_armGLM", (DL_FUNC) &_armb_tune_armGLM, 18},
    {"_armb_test_glm", (DL_FUNC) &_armb_test_glm, 5},
    {"_armb_shuffleRows", (DL_FUNC) &_armb_shuffleRows, 2},
    {"_armb_shuffleVec", (DL_FUNC) &_armb_shuffleVec, 2},
    {"_armb_resampleN", (DL_FUNC) &_armb_resampleN, 2},
    {"_armb_sliceVec", (DL_FUNC) &_armb_sliceVec, 2},
    {"_armb_sliceMat", (DL_FUNC) &_armb_sliceMat, 2},
    {"_armb_subsetIVec", (DL_FUNC) &_armb_subsetIVec, 3},
    {"_armb_shuffleIVec", (DL_FUNC) &_armb_shuffleIVec, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_armb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
