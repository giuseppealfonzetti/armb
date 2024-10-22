#' ARMB nonparametric bootstrap
#'
#' @description
#' Fast nonparametric bootstrap for glm parameters using stochastic approximations.
#'
#' @param DATA Dataset.
#' @param R Number of bootstrap samples.
#' @param SEED Argument passed to `set.seed()` for reproducibility purposes in bootstrap samples.
#' @param NCORES Number of cores used via the \link[pbapply]{pblapply} function.
#'
#' @details Returns a tibble with classes `bootstraps`, `rset`, `tbl_df`, `tbl`, `data.frame`.
#' See \link[rsample]{bootstraps} for further details. The tibble include a column
#' with bootstrap estimates obtained using \link[sgd]{sgd} algorithms.
#'
#' @references
#' Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
#' approximation by averaging. \emph{SIAM Journal on Control and Optimization},
#' 30(4):838-855, 1992.
#'
#' @importFrom boot boot
#' @export
armb <- function(Y, X, FAMILY, NSIM, MLE, TUNE_STEP, TUNE_GAMMA, NCORES, SEED, ARM_CONTROL, TUNE_CONTROL, VERBOSE = FALSE){
  start <- Sys.time()

  res <- list()
  if(TUNE_STEP){
    if(VERBOSE)cat("Tuning the stepsize... ")
    tune <- tune_armGLM(
      Y = as.numeric(Y),
      X = X,
      FAMILY = FAMILY$family,
      LINK = FAMILY$link,
      THETA0 = rep(0, length(MLE)),
      MAXT = TUNE_CONTROL$MAXT,
      BURN = ARM_CONTROL$BURN,
      BATCH = ARM_CONTROL$BATCH,
      STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
      SCALE = TUNE_CONTROL$SCALE,
      MAXA = TUNE_CONTROL$MAXA,
      PAR1 = ARM_CONTROL$PAR1,
      PAR2 = ARM_CONTROL$PAR2,
      PAR3 = ARM_CONTROL$PAR3,
      AUTO_STOP = TUNE_CONTROL$AUTO,
      VERBOSE = TUNE_CONTROL$VERBOSE,
      SEED = ARM_CONTROL$SEED,
      SKIP_PRINT = 0
    )
    res$tune <- tune
    ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$devresids)][1]
    if(VERBOSE)cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
  }

  if(TUNE_GAMMA){
    if(VERBOSE)cat("Tuning gamma...        ")
    tune <- armGLM(Y = as.numeric(Y),
                   X = X,
                   FAMILY = FAMILY$family,
                   LINK = FAMILY$link,
                   THETA0 = MLE,
                   MAXT = ARM_CONTROL$MAXT,
                   BURN = ARM_CONTROL$BURN,
                   BATCH = ARM_CONTROL$BATCH,
                   STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
                   PAR1 = ARM_CONTROL$PAR1,
                   PAR2 = ARM_CONTROL$PAR2,
                   PAR3 = ARM_CONTROL$PAR3,
                   PATH_WINDOW = ARM_CONTROL$PATH_WINDOW,
                   VERBOSE_WINDOW = ARM_CONTROL$VERBOSE_WINDOW,
                   VERBOSE = TUNE_CONTROL$VERBOSE,
                   SEED = ARM_CONTROL$SEED,
                   CONV_WINDOW = TUNE_CONTROL$CONV_WINDOW,
                   CONV_CHECK = TRUE,
                   TOL = TUNE_CONTROL$CONV_TOL)
    ARM_CONTROL$MAXT <- tune$last_iter
    if(VERBOSE)cat("| Value chosen:", round(logb(tune$last_iter, length(Y)), 4), "\n")

  }

  sgdFun <- function(DATA,  IDX, THETA0, CONTROL){
    fit <- armGLM(Y = as.numeric(DATA[IDX,1]),
                  X = DATA[IDX,-1],
                  FAMILY = FAMILY$family,
                  LINK = FAMILY$link,
                  THETA0 = THETA0,
                  MAXT = CONTROL$MAXT,
                  BURN = CONTROL$BURN,
                  BATCH = CONTROL$BATCH,
                  STEPSIZE0 = CONTROL$STEPSIZE0,
                  PAR1 = CONTROL$PAR1,
                  PAR2 = CONTROL$PAR2,
                  PAR3 = CONTROL$PAR3,
                  PATH_WINDOW = CONTROL$PATH_WINDOW,
                  VERBOSE_WINDOW = CONTROL$VERBOSE_WINDOW,
                  VERBOSE = CONTROL$VERBOSE,
                  SEED = CONTROL$SEED
    )

    out <- fit$avtheta
    return(out)
  }

  if(VERBOSE)cat("Running ", NSIM, " ARM chains... ")
  armbt <- boot(
    data = cbind(Y, X),
    statistic = sgdFun,
    R = NSIM,
    ncpus = NCORES,
    THETA0 = MLE,#
    CONTROL = ARM_CONTROL)

  res$step0 <- ARM_CONTROL$STEPSIZE0
  res$gamma <- logb(ARM_CONTROL$MAXT, length(Y))
  res$time <- difftime(Sys.time(),start, units = 'secs')
  res$pars <- armbt$t
  if(VERBOSE)cat("Done!\n")

  return(res)
}

