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
fastBoot <- function(DATA, R, FORMULA, MODEL = 'glm',
                     MODEL.CONTROL = list(),
                     SGD.CONTROL = list(method = 'asgd', npasses = 1, pass = TRUE),
                     FR = .1,
                     SEED = 123,
                     NCORES = 1,
                     TUNEFLAG = T,
                     TUNE.CONTROL = list(FR_N = 1, MAX_ATTEMPTS = 10, TOL = 1e-5)){
  if(!(SGD.CONTROL[['method']] %in% c('asgd', 'ai-sgd', 'numeric'))) stop('The stochastic approximation must be averaged.\nOnly `asgd` and `ai-sgd` methods are available.')
  SGD.CONTROL[['pass']] <- TRUE
  if(is.null(SGD.CONTROL[['start']])){SGD.CONTROL[['start']] <- rep(0, ncol(model.matrix(object = FORMULA, data = DATA)))}

  set.seed(SEED)

  tune <- list()
  if(TUNEFLAG){
    tune <- tuneStep(
      FR_N = TUNE.CONTROL[['FR_N']],
      MAX_ATTEMPTS = TUNE.CONTROL[['MAX_ATTEMPTS']],
      MOD.CTRL = MODEL.CONTROL,
      SGD.CTRL = SGD.CONTROL,
      DATA = DATA,
      FORMULA = FORMULA,
      MODEL = MODEL,
      TOL = TUNE.CONTROL[['TOL']]
    )

    SGD.CONTROL[['lr.control']][['gamma']] <- tune$stepsize[which(tune$nll == min(tune$nll))]
  }

  sgdFun <- function(D, IDX,FORMULA, MODEL, MODEL.CONTROL, SGD.CONTROL, FR){
    sgd.ctrl <- SGD.CONTROL
    mod <- sgd::sgd(formula = FORMULA, model = MODEL,
                    model.control = MODEL.CONTROL, sgd.control = sgd.ctrl,
                    data = D[IDX[1:(FR*nrow(D))], ])

    out <- as.numeric(mod$coefficients)
    return(out)
  }

  bt <- boot::boot(
    data = DATA,
    statistic = sgdFun,
    R = R,
    parallel = 'multicore',
    ncpus = NCORES,
    FORMULA = FORMULA,
    MODEL = MODEL,
    FR = FR,
    MODEL.CONTROL = MODEL.CONTROL,
    SGD.CONTROL = SGD.CONTROL)

  return(list(tune = tune, bt = bt))
}
