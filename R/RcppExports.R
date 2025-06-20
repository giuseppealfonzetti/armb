# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Compute nll and ngr for logistic regression
#'
#' @description Used for internal testing
#'
#' @param Y response
#' @param X design matrix
#' @param THETA parameter vector
#'
#' @returns Provides a list with the negative log likelihood and its gradient
#'
#' @export
logistic_wrapper <- function(Y, X, THETA) {
    .Call(`_armb_logistic_wrapper`, Y, X, THETA)
}

logsumexp <- function(x) {
    .Call(`_armb_logsumexp`, x)
}

innerProduct <- function(x, y) {
    .Call(`_armb_innerProduct`, x, y)
}

likAGHi <- function(beta, logsigma, list_x, list_y, list_d, niter, ws, z, i) {
    .Call(`_armb_likAGHi`, beta, logsigma, list_x, list_y, list_d, niter, ws, z, i)
}

grAGHi <- function(betavec, logsigma, list_x, list_y, list_d, niter, ws, z, i) {
    .Call(`_armb_grAGHi`, betavec, logsigma, list_x, list_y, list_d, niter, ws, z, i)
}

likAGH <- function(betavec, sigma, list_x, list_y, list_d, niter, ws, z) {
    .Call(`_armb_likAGH`, betavec, sigma, list_x, list_y, list_d, niter, ws, z)
}

grAGH <- function(beta, sigma, list_x, list_y, list_d, niter, ws, z) {
    .Call(`_armb_grAGH`, beta, sigma, list_x, list_y, list_d, niter, ws, z)
}

#' @export
armLOGRI <- function(LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, VERBOSE_WINDOW, PATH_WINDOW, SEED, VERBOSE, CONV_WINDOW = 1000L, CONV_CHECK = FALSE, TOL = 1e-5) {
    .Call(`_armb_armLOGRI`, LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, VERBOSE_WINDOW, PATH_WINDOW, SEED, VERBOSE, CONV_WINDOW, CONV_CHECK, TOL)
}

#' @export
tune_armLOGRI <- function(LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE) {
    .Call(`_armb_tune_armLOGRI`, LIST_X, LIST_Y, LIST_D, THETA0, AGH_NITER, WS, Z, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE)
}

#' Fit logistic regression via ARM
#'
#' @param Y response
#' @param X design matrix
#' @param THETA0 starting values
#' @param MAXT max number of iterations
#' @param MAXE max number of epochs
#' @param BURN burn-in period
#' @param BATCH mini-natch dimension
#' @param STEPSIZE0 initial stepsize
#' @param PAR1 par1
#' @param PAR2 par2
#' @param PAR3 par3
#' @param SKIP_PRINT How many iterations to skip before printing diagnostics
#' @param VERBOSE verbose output
#'
#' @export
armLR2 <- function(Y, X, THETA0, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, STORE, SKIP_PRINT, SEED, VERBOSE) {
    .Call(`_armb_armLR2`, Y, X, THETA0, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, STORE, SKIP_PRINT, SEED, VERBOSE)
}

#' tune logistic regression via ARM
#'
#' @param Y response
#' @param X design matrix
#' @param THETA0 starting values
#' @param MAXT max number of iterations
#' @param MAXE max number of epochs
#' @param BURN burn-in period
#' @param BATCH mini-natch dimension
#' @param STEPSIZE0 initial stepsize
#' @param SCALE multiplying factor
#' @param MAXA maximum number of attempts
#' @param PAR1 par1
#' @param PAR2 par2
#' @param PAR3 par3
#' @param SKIP_PRINT How many iterations to skip before printing diagnostics
#' @param VERBOSE verbose output
#'
#' @export
tune_armLR <- function(Y, X, THETA0, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE) {
    .Call(`_armb_tune_armLR`, Y, X, THETA0, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE)
}

#' @export
armbLR <- function(Y, X, THETA0, R, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, STORE, SKIP_PRINT, SEED, VERBOSE) {
    .Call(`_armb_armbLR`, Y, X, THETA0, R, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, STORE, SKIP_PRINT, SEED, VERBOSE)
}

#' @export
armGLM <- function(Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, VERBOSE_WINDOW, PATH_WINDOW, SEED, VERBOSE, CONV_WINDOW = 1000L, CONV_CHECK = FALSE, TOL = 1e-5) {
    .Call(`_armb_armGLM`, Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, STEPSIZE0, PAR1, PAR2, PAR3, VERBOSE_WINDOW, PATH_WINDOW, SEED, VERBOSE, CONV_WINDOW, CONV_CHECK, TOL)
}

#' @export
tune_armGLM <- function(Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE) {
    .Call(`_armb_tune_armGLM`, Y, X, FAMILY, LINK, THETA0, MAXT, BURN, BATCH, STEPSIZE0, SCALE, MAXA, PAR1, PAR2, PAR3, AUTO_STOP, SKIP_PRINT, SEED, VERBOSE)
}

#' Compute nll and ngr for logistic regression
#'
#' @description Used for internal testing
#'
#' @param Y response
#' @param X design matrix
#' @param THETA parameter vector
#'
#' @returns Provides a list with the negative log likelihood and its gradient
#'
#' @export
test_glm <- function(Y, X, FAMILY, LINK, THETA) {
    .Call(`_armb_test_glm`, Y, X, FAMILY, LINK, THETA)
}

#' Randomly shuffle rows of a matrix
#'
#' @param X matrix
#' @param SEED seed for rng
#'
#' @export
shuffleRows <- function(X, SEED) {
    .Call(`_armb_shuffleRows`, X, SEED)
}

#' Randomly shuffle elements of a vector
#'
#' @param X vector
#' @param SEED seed for rng
#'
#' @export
shuffleVec <- function(X, SEED) {
    .Call(`_armb_shuffleVec`, X, SEED)
}

#' Resample sequence
#'
#' Resample with replacement the sequence from 0 to N-1
#'
#' @param N length of the sequence
#' @param SEED seed for rng
#'
#' @export
resampleN <- function(N, SEED) {
    .Call(`_armb_resampleN`, N, SEED)
}

#' Resample sequence
#'
#' Resample with replacement the sequence from 0 to N-1
#'
#' @param N length of the sequence
#' @param SEED seed for rng
#'
#' @export
sliceVec <- function(SLICE_IDX, X) {
    .Call(`_armb_sliceVec`, SLICE_IDX, X)
}

#' Resample sequence
#'
#' Resample with replacement the sequence from 0 to N-1
#'
#' @param N length of the sequence
#' @param SEED seed for rng
#'
#' @export
sliceMat <- function(SLICE_IDX, X) {
    .Call(`_armb_sliceMat`, SLICE_IDX, X)
}

#' @export
subsetIVec <- function(X, START, LEN) {
    .Call(`_armb_subsetIVec`, X, START, LEN)
}

#' @export
shuffleIVec <- function(X, SEED) {
    .Call(`_armb_shuffleIVec`, X, SEED)
}

