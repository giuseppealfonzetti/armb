# #' ARMB
# #'
# #' @description
# #' Fast nonparametric bootstrap for glm parameters using stochastic approximations.
# #'
# #' @param DATA Dataset.
# #' @param R Number of bootstrap samples.
# #' @param SEED Argument passed to `set.seed()` for reproducibility purposes in bootstrap samples.
# #' @param NCORES Number of cores used via the \link[pbapply]{pblapply} function.
# #'
# #' @details Returns a tibble with classes `bootstraps`, `rset`, `tbl_df`, `tbl`, `data.frame`.
# #' See \link[rsample]{bootstraps} for further details. The tibble include a column
# #' with bootstrap estimates obtained using \link[sgd]{sgd} algorithms.
# #'
# #' @references
# #' Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
# #' approximation by averaging. \emph{SIAM Journal on Control and Optimization},
# #' 30(4):838-855, 1992.
# #'
# #' @importFrom boot boot
# #' @export
# armb <- function(
#   Y,
#   X,
#   FAMILY,
#   NSIM,
#   MLE,
#   TUNE_STEP,
#   TUNE_GAMMA,
#   NCORES,
#   SEED,
#   ARM_CONTROL,
#   TUNE_CONTROL,
#   VERBOSE = FALSE
# ) {
#   start <- Sys.time()

#   res <- list()
#   if (TUNE_STEP) {
#     if (VERBOSE) {
#       cat("Tuning the stepsize... ")
#     }

#     tune <- tune_armGLM(
#       Y = as.numeric(Y),
#       X = X,
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = rep(0, length(MLE)),
#       MAXT = TUNE_CONTROL$MAXT,
#       BURN = ARM_CONTROL$BURN,
#       BATCH = ARM_CONTROL$BATCH,
#       STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#       SCALE = TUNE_CONTROL$SCALE,
#       MAXA = TUNE_CONTROL$MAXA,
#       PAR1 = ARM_CONTROL$PAR1,
#       PAR2 = ARM_CONTROL$PAR2,
#       PAR3 = ARM_CONTROL$PAR3,
#       AUTO_STOP = TUNE_CONTROL$AUTO,
#       VERBOSE = TUNE_CONTROL$VERBOSE,
#       SEED = ARM_CONTROL$SEED,
#       SKIP_PRINT = 0
#     )
#     res$tune <- tune
#     ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$devresids)][1]
#     if (VERBOSE) cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
#   }

#   if (TUNE_GAMMA) {
#     if (VERBOSE) {
#       cat("Tuning gamma...        ")
#     }
#     tune <- armGLM(
#       Y = as.numeric(Y),
#       X = X,
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = MLE,
#       MAXT = ARM_CONTROL$MAXT,
#       BURN = ARM_CONTROL$BURN,
#       BATCH = ARM_CONTROL$BATCH,
#       STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#       PAR1 = ARM_CONTROL$PAR1,
#       PAR2 = ARM_CONTROL$PAR2,
#       PAR3 = ARM_CONTROL$PAR3,
#       PATH_WINDOW = ARM_CONTROL$PATH_WINDOW,
#       VERBOSE_WINDOW = ARM_CONTROL$VERBOSE_WINDOW,
#       VERBOSE = TUNE_CONTROL$VERBOSE,
#       SEED = ARM_CONTROL$SEED,
#       CONV_WINDOW = TUNE_CONTROL$CONV_WINDOW,
#       CONV_CHECK = TRUE,
#       TOL = TUNE_CONTROL$CONV_TOL
#     )
#     ARM_CONTROL$MAXT <- tune$last_iter
#     if (VERBOSE) {
#       cat(
#         "| Value chosen:",
#         round(logb(tune$last_iter - ARM_CONTROL$BURN, length(Y)), 4),
#         "\n"
#       )
#     }
#   }

#   sgdFun <- function(DATA, IDX, THETA0, CONTROL) {
#     fit <- armGLM(
#       Y = as.numeric(DATA[IDX, 1]),
#       X = DATA[IDX, -1],
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = THETA0,
#       MAXT = CONTROL$MAXT,
#       BURN = CONTROL$BURN,
#       BATCH = CONTROL$BATCH,
#       STEPSIZE0 = CONTROL$STEPSIZE0,
#       PAR1 = CONTROL$PAR1,
#       PAR2 = CONTROL$PAR2,
#       PAR3 = CONTROL$PAR3,
#       PATH_WINDOW = CONTROL$PATH_WINDOW,
#       VERBOSE_WINDOW = CONTROL$VERBOSE_WINDOW,
#       VERBOSE = CONTROL$VERBOSE,
#       SEED = CONTROL$SEED
#     )

#     out <- fit$avtheta
#     return(out)
#   }

#   if (VERBOSE) {
#     cat("Running ", NSIM, " ARM chains... ")
#   }
#   armbt <- boot(
#     data = cbind(Y, X),
#     statistic = sgdFun,
#     R = NSIM,
#     ncpus = NCORES,
#     THETA0 = MLE, #
#     CONTROL = ARM_CONTROL
#   )

#   res$step0 <- ARM_CONTROL$STEPSIZE0
#   res$gamma <- logb(ARM_CONTROL$MAXT - ARM_CONTROL$BURN, length(Y))
#   res$time <- difftime(Sys.time(), start, units = 'secs')
#   res$pars <- armbt$t
#   if (VERBOSE) {
#     cat("Done!\n")
#   }

#   return(res)
# }

# #' @importFrom boot boot
# #' @importFrom statmod gauss.quad
# #' @export
# armb2 <- function(
#   Y,
#   X,
#   FAMILY,
#   NSIM,
#   MLE,
#   FRML = NULL,
#   TUNE_STEP,
#   TUNE_GAMMA,
#   NCORES,
#   SEED,
#   ARM_CONTROL,
#   TUNE_CONTROL,
#   VERBOSE = FALSE,
#   NGH = 15
# ) {
#   start <- Sys.time()

#   data_list <- list()
#   if (FAMILY == "LOGRI") {
#     obj.gh <- statmod::gauss.quad(NGH, "hermite")
#     ws <- obj.gh$weights * exp(obj.gh$nodes^2)
#     df <- cbind(Y, X)
#     listx <- lapply(
#       by(model.matrix(as.formula(FRML), data = df) * 1.0, df$g, identity),
#       as.matrix
#     )
#     listy <- split(Y * 1.0, df$g)
#     listd <- split(df$size * 1.0, df$g)
#     data_list <- list(
#       x = listx,
#       y = listy,
#       den = listd,
#       niter = 10,
#       ws = ws,
#       z = obj.gh$nodes
#     )
#   }

#   res <- list()
#   if (TUNE_STEP) {
#     if (VERBOSE) {
#       cat("Tuning the stepsize... ")
#     }
#     if (FAMILY == "LOGRI") {
#       tune <- tune_armLOGRI(
#         LIST_X = data_list$x,
#         LIST_Y = data_list$y,
#         LIST_D = data_list$den,
#         THETA0 = MLE,
#         AGH_NITER = data_list$niter,
#         WS = data_list$ws,
#         Z = data_list$z,
#         MAXT = TUNE_CONTROL$MAXT,
#         BURN = ARM_CONTROL$BURN,
#         BATCH = ARM_CONTROL$BATCH,
#         STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#         SCALE = TUNE_CONTROL$SCALE,
#         MAXA = TUNE_CONTROL$MAXA,
#         PAR1 = ARM_CONTROL$PAR1,
#         PAR2 = ARM_CONTROL$PAR2,
#         PAR3 = ARM_CONTROL$PAR3,
#         AUTO_STOP = TUNE_CONTROL$AUTO,
#         VERBOSE = TUNE_CONTROL$VERBOSE,
#         SEED = ARM_CONTROL$SEED,
#         SKIP_PRINT = 0
#       )
#       res$tune <- tune
#       ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$devresids)][1]
#       if (VERBOSE) cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
#     } else {
#       tune <- tune_armGLM(
#         Y = as.numeric(Y),
#         X = X,
#         FAMILY = FAMILY$family,
#         LINK = FAMILY$link,
#         THETA0 = rep(0, length(MLE)),
#         MAXT = TUNE_CONTROL$MAXT,
#         BURN = ARM_CONTROL$BURN,
#         BATCH = ARM_CONTROL$BATCH,
#         STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#         SCALE = TUNE_CONTROL$SCALE,
#         MAXA = TUNE_CONTROL$MAXA,
#         PAR1 = ARM_CONTROL$PAR1,
#         PAR2 = ARM_CONTROL$PAR2,
#         PAR3 = ARM_CONTROL$PAR3,
#         AUTO_STOP = TUNE_CONTROL$AUTO,
#         VERBOSE = TUNE_CONTROL$VERBOSE,
#         SEED = ARM_CONTROL$SEED,
#         SKIP_PRINT = 0
#       )
#       res$tune <- tune
#       ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$devresids)][1]
#       if (VERBOSE) cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
#     }
#   }

#   if (TUNE_GAMMA) {
#     if (VERBOSE) {
#       cat("Tuning gamma...        ")
#     }
#     tune <- armGLM(
#       Y = as.numeric(Y),
#       X = X,
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = MLE,
#       MAXT = ARM_CONTROL$MAXT,
#       BURN = ARM_CONTROL$BURN,
#       BATCH = ARM_CONTROL$BATCH,
#       STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#       PAR1 = ARM_CONTROL$PAR1,
#       PAR2 = ARM_CONTROL$PAR2,
#       PAR3 = ARM_CONTROL$PAR3,
#       PATH_WINDOW = ARM_CONTROL$PATH_WINDOW,
#       VERBOSE_WINDOW = ARM_CONTROL$VERBOSE_WINDOW,
#       VERBOSE = TUNE_CONTROL$VERBOSE,
#       SEED = ARM_CONTROL$SEED,
#       CONV_WINDOW = TUNE_CONTROL$CONV_WINDOW,
#       CONV_CHECK = TRUE,
#       TOL = TUNE_CONTROL$CONV_TOL
#     )
#     ARM_CONTROL$MAXT <- tune$last_iter
#     if (VERBOSE) {
#       cat(
#         "| Value chosen:",
#         round(logb(tune$last_iter - ARM_CONTROL$BURN, length(Y)), 4),
#         "\n"
#       )
#     }
#   }

#   armbt <- NULL
#   sgdFun <- function(DATA, IDX, THETA0, CONTROL) {
#     if (FAMILY == "LOGRI") {
#       fit <- armLOGRI(
#         LIST_X = DATA$x[IDX],
#         LIST_Y = DATA$y[IDX],
#         LIST_D = DATA$den[IDX],
#         THETA0 = THETA0,
#         AGH_NITER = DATA$niter,
#         WS = DATA$ws,
#         Z = DATA$z,
#         MAXT = CONTROL$MAXT,
#         BURN = CONTROL$BURN,
#         BATCH = CONTROL$BATCH,
#         STEPSIZE0 = CONTROL$STEPSIZE0,
#         PAR1 = CONTROL$PAR1,
#         PAR2 = CONTROL$PAR2,
#         PAR3 = CONTROL$PAR3,
#         PATH_WINDOW = CONTROL$PATH_WINDOW,
#         VERBOSE_WINDOW = CONTROL$VERBOSE_WINDOW,
#         VERBOSE = CONTROL$VERBOSE,
#         SEED = CONTROL$SEED
#       )
#     } else {
#       fit <- armGLM(
#         Y = as.numeric(DATA[IDX, 1]),
#         X = DATA[IDX, -1],
#         FAMILY = FAMILY$family,
#         LINK = FAMILY$link,
#         THETA0 = THETA0,
#         MAXT = CONTROL$MAXT,
#         BURN = CONTROL$BURN,
#         BATCH = CONTROL$BATCH,
#         STEPSIZE0 = CONTROL$STEPSIZE0,
#         PAR1 = CONTROL$PAR1,
#         PAR2 = CONTROL$PAR2,
#         PAR3 = CONTROL$PAR3,
#         PATH_WINDOW = CONTROL$PATH_WINDOW,
#         VERBOSE_WINDOW = CONTROL$VERBOSE_WINDOW,
#         VERBOSE = CONTROL$VERBOSE,
#         SEED = CONTROL$SEED
#       )
#     }

#     out <- fit$avtheta
#     return(out)
#   }

#   if (VERBOSE) {
#     cat("Running ", NSIM, " ARM chains... ")
#   }
#   armbt <- list()
#   if (FAMILY == "LOGRI") {
#     armbt$t <- t(sapply(
#       1:NSIM,
#       function(x) {
#         set.seed(x)
#         sgdFun(
#           DATA = data_list,
#           IDX = sample(
#             1:length(data_list$x),
#             length(data_list$x),
#             replace = TRUE
#           ),
#           THETA0 = MLE,
#           CONTROL = ARM_CONTROL
#         )
#       }
#     ))
#   } else {
#     armbt <- boot(
#       data = if (FAMILY == "LOGRI") {
#         data_list
#       } else {
#         cbind(Y, X)
#       },
#       statistic = sgdFun,
#       R = NSIM,
#       ncpus = NCORES,
#       THETA0 = MLE, #
#       CONTROL = ARM_CONTROL
#     )
#   }

#   res$step0 <- ARM_CONTROL$STEPSIZE0
#   res$gamma <- logb(ARM_CONTROL$MAXT - ARM_CONTROL$BURN, length(Y))
#   res$time <- difftime(Sys.time(), start, units = 'secs')
#   res$pars <- armbt$t
#   if (VERBOSE) {
#     cat("Done!\n")
#   }

#   return(res)
# }

# #' ARMB
# #'
# #' @description
# #' Fast nonparametric bootstrap for glm parameters using stochastic approximations.
# #'
# #' @param DATA Dataset.
# #' @param R Number of bootstrap samples.
# #' @param SEED Argument passed to `set.seed()` for reproducibility purposes in bootstrap samples.
# #' @param NCORES Number of cores used via the \link[pbapply]{pblapply} function.
# #'
# #' @details Returns a tibble with classes `bootstraps`, `rset`, `tbl_df`, `tbl`, `data.frame`.
# #' See \link[rsample]{bootstraps} for further details. The tibble include a column
# #' with bootstrap estimates obtained using \link[sgd]{sgd} algorithms.
# #'
# #' @references
# #' Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
# #' approximation by averaging. \emph{SIAM Journal on Control and Optimization},
# #' 30(4):838-855, 1992.
# #'
# #' @importFrom boot boot
# #' @export
# armb3 <- function(
#   Y,
#   X,
#   FAMILY,
#   NSIM,
#   MLE,
#   TUNE_STEP,
#   TUNE_GAMMA,
#   NCORES,
#   SEED,
#   ARM_CONTROL,
#   TUNE_CONTROL,
#   VERBOSE = FALSE
# ) {
#   start <- Sys.time()

#   res <- list()
#   if (TUNE_STEP) {
#     if (VERBOSE) {
#       cat("Tuning the stepsize... ")
#     }
#     set.seed(SEED)
#     resample_ids <- sample(1:length(Y), length(Y), replace = TRUE)
#     tune <- tune_armGLM2(
#       Y = as.numeric(Y[resample_ids]),
#       X = X[resample_ids, ],
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = MLE, #+
#       #sapply(MLE, function(x) rnorm(1, 0, abs(x))),
#       MAXT = TUNE_CONTROL$MAXT,
#       BURN = ARM_CONTROL$BURN,
#       BATCH = ARM_CONTROL$BATCH,
#       STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#       SCALE = TUNE_CONTROL$SCALE,
#       MAXA = TUNE_CONTROL$MAXA,
#       PAR1 = ARM_CONTROL$PAR1,
#       PAR2 = ARM_CONTROL$PAR2,
#       PAR3 = ARM_CONTROL$PAR3,
#       AUTO_STOP = TUNE_CONTROL$AUTO,
#       VERBOSE = TUNE_CONTROL$VERBOSE,
#       SEED = ARM_CONTROL$SEED,
#       SKIP_PRINT = 0
#     )
#     res$tune <- tune
#     ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$devresids)][1]
#     if (VERBOSE) cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
#   }

#   if (TUNE_GAMMA) {
#     if (VERBOSE) {
#       cat("Tuning gamma...        ")
#     }
#     tune <- armGLM(
#       Y = as.numeric(Y),
#       X = X,
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = MLE,
#       MAXT = ARM_CONTROL$MAXT,
#       BURN = ARM_CONTROL$BURN,
#       BATCH = ARM_CONTROL$BATCH,
#       STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
#       PAR1 = ARM_CONTROL$PAR1,
#       PAR2 = ARM_CONTROL$PAR2,
#       PAR3 = ARM_CONTROL$PAR3,
#       PATH_WINDOW = ARM_CONTROL$PATH_WINDOW,
#       VERBOSE_WINDOW = ARM_CONTROL$VERBOSE_WINDOW,
#       VERBOSE = TUNE_CONTROL$VERBOSE,
#       SEED = ARM_CONTROL$SEED,
#       CONV_WINDOW = TUNE_CONTROL$CONV_WINDOW,
#       CONV_CHECK = TRUE,
#       TOL = TUNE_CONTROL$CONV_TOL
#     )
#     ARM_CONTROL$MAXT <- tune$last_iter
#     if (VERBOSE) {
#       cat(
#         "| Value chosen:",
#         round(logb(tune$last_iter - ARM_CONTROL$BURN, length(Y)), 4),
#         "\n"
#       )
#     }
#   }

#   sgdFun <- function(DATA, IDX, THETA0, CONTROL) {
#     fit <- armGLM2(
#       Y = as.numeric(DATA[IDX, 1]),
#       X = DATA[IDX, -1],
#       FAMILY = FAMILY$family,
#       LINK = FAMILY$link,
#       THETA0 = THETA0, #+
#       #sapply(THETA0, function(x) rnorm(1, 0, abs(x))),
#       MAXT = CONTROL$MAXT,
#       BURN = CONTROL$BURN,
#       BATCH = CONTROL$BATCH,
#       STEPSIZE0 = CONTROL$STEPSIZE0,
#       PAR1 = CONTROL$PAR1,
#       PAR2 = CONTROL$PAR2,
#       PAR3 = CONTROL$PAR3,
#       PATH_WINDOW = CONTROL$PATH_WINDOW,
#       VERBOSE_WINDOW = CONTROL$VERBOSE_WINDOW,
#       VERBOSE = CONTROL$VERBOSE,
#       SEED = CONTROL$SEED
#     )

#     out <- fit$avtheta
#     return(out)
#   }

#   if (VERBOSE) {
#     cat("Running ", NSIM, " ARM chains... ")
#   }
#   armbt <- boot(
#     data = cbind(Y, X),
#     statistic = sgdFun,
#     R = NSIM,
#     ncpus = NCORES,
#     THETA0 = MLE, #
#     CONTROL = ARM_CONTROL
#   )

#   res$step0 <- ARM_CONTROL$STEPSIZE0
#   res$gamma <- logb(ARM_CONTROL$MAXT - ARM_CONTROL$BURN, length(Y))
#   res$time <- difftime(Sys.time(), start, units = 'secs')
#   res$pars <- armbt$t
#   if (VERBOSE) {
#     cat("Done!\n")
#   }

#   return(res)
# }

#' ARMB
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
armb4 <- function(
  Y,
  X,
  FAMILY,
  NSIM,
  MLE,
  TUNE_STEP,
  NCORES,
  SEED,
  ARM_CONTROL,
  TUNE_CONTROL,
  VERBOSE = FALSE,
  TYPE = c("GLM", "GLMM"),
  FRML = NULL,
  NGH = 9
) {
  TYPE <- match.arg(TYPE)
  if (TYPE == "GLMM" & (FAMILY$family != "binomial" | FAMILY$link != "logit")) {
    stop("Family not implemented for GLMMs")
  }

  start <- Sys.time()

  data_list <- list()
  if (TYPE == "GLMM") {
    obj.gh <- statmod::gauss.quad(NGH, "hermite")
    ws <- obj.gh$weights * exp(obj.gh$nodes^2)
    df <- cbind(Y, X)
    listx <- lapply(
      by(model.matrix(as.formula(FRML), data = df) * 1.0, df$g, identity),
      as.matrix
    )
    listy <- split(Y * 1.0, df$g)
    listd <- split(df$size * 1.0, df$g)
    data_list <- list(
      x = listx,
      y = listy,
      den = listd,
      niter = 10,
      ws = ws,
      z = obj.gh$nodes
    )
  }

  res <- list()

  if (TUNE_STEP) {
    if (VERBOSE) {
      cat("Tuning the stepsize... ")
    }

    if (TYPE == "GLM") {
      set.seed(SEED)
      resample_ids <- sample(1:length(Y), length(Y), replace = TRUE)
      tune <- tune_armGLM3(
        Y = as.numeric(Y[resample_ids]),
        X = X[resample_ids, ],
        FAMILY = FAMILY$family,
        LINK = FAMILY$link,
        THETA0 = MLE,
        LENGTH = TUNE_CONTROL$LENGTH,
        BURN = TUNE_CONTROL$BURN,
        STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
        SCALE = TUNE_CONTROL$SCALE,
        MAXA = TUNE_CONTROL$MAXA,
        AUTO_STOP = TUNE_CONTROL$AUTO,
        VERBOSE = TUNE_CONTROL$VERBOSE,
        SEED = ARM_CONTROL$SEED,
        CONV_CHECK = ARM_CONTROL$CONV_CHECK,
        CONV_WINDOW = ARM_CONTROL$CONV_WINDOW,
        TOL = ARM_CONTROL$TOL
      )

      res$tune <- tune
      ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$nlls)][1]
      if (VERBOSE) cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
    } else {
      set.seed(SEED)
      resample_idx <- sample(
        1:length(data_list$x),
        length(data_list$x),
        replace = TRUE
      )
      tune <- tune_armLOGRI2(
        LIST_X = data_list$x[resample_idx],
        LIST_Y = data_list$y[resample_idx],
        LIST_D = data_list$den[resample_idx],
        THETA0 = MLE,
        AGH_NITER = data_list$niter,
        WS = data_list$ws,
        Z = data_list$z,
        LENGTH = TUNE_CONTROL$LENGTH,
        BURN = TUNE_CONTROL$BURN,
        STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
        SCALE = TUNE_CONTROL$SCALE,
        MAXA = TUNE_CONTROL$MAXA,
        AUTO_STOP = TUNE_CONTROL$AUTO,
        VERBOSE = TUNE_CONTROL$VERBOSE,
        SEED = ARM_CONTROL$SEED,
        CONV_CHECK = ARM_CONTROL$CONV_CHECK,
        CONV_WINDOW = ARM_CONTROL$CONV_WINDOW,
        TOL = ARM_CONTROL$TOL
      )
      res$tune <- tune
      ARM_CONTROL$STEPSIZE0 <- tune$stepsizes[which.min(tune$nlls)][1]
      if (VERBOSE) cat("| Value chosen:", round(ARM_CONTROL$STEPSIZE0, 4), "\n")
    }
  }

  if (VERBOSE) {
    cat("Running ", NSIM, " ARM chains... ")
  }

  chains <- list()
  if (TYPE == "GLM") {
    chains <- pbapply::pblapply(
      1:NSIM,
      function(id) {
        set.seed(id)
        resample_idx <- sample(1:nrow(X), nrow(X), replace = TRUE)

        chain.id <- 0
        chain.id <- armGLM3(
          Y = Y[resample_idx],
          X = X[resample_idx, ],
          FAMILY = FAMILY$family,
          LINK = FAMILY$link,
          THETA0 = MLE,
          LENGTH = length(Y),
          BURN = ARM_CONTROL$BURN,
          STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
          TRIM = ARM_CONTROL$TRIM,
          VERBOSE = ARM_CONTROL$VERBOSE,
          SEED = ARM_CONTROL$SEED,
          CONV_CHECK = ARM_CONTROL$CONV_CHECK,
          CONV_WINDOW = ARM_CONTROL$CONV_WINDOW,
          TOL = ARM_CONTROL$TOL
        )
        return(chain.id)
      },
      cl = NCORES
    )
  } else {
    chains <- pbapply::pblapply(
      1:NSIM,
      function(id) {
        set.seed(id)
        resample_idx <- sample(
          1:length(data_list$x),
          length(data_list$x),
          replace = TRUE
        )
        chain.id <- 0
        chain.id <- armLOGRI2(
          LIST_X = data_list$x[resample_idx],
          LIST_Y = data_list$y[resample_idx],
          LIST_D = data_list$den[resample_idx],
          THETA0 = MLE,
          AGH_NITER = data_list$niter,
          WS = data_list$ws,
          Z = data_list$z,
          LENGTH = length(data_list$y),
          BURN = ARM_CONTROL$BURN,
          STEPSIZE0 = ARM_CONTROL$STEPSIZE0,
          TRIM = ARM_CONTROL$TRIM,
          VERBOSE = ARM_CONTROL$VERBOSE,
          SEED = ARM_CONTROL$SEED,
          CONV_CHECK = ARM_CONTROL$CONV_CHECK,
          CONV_WINDOW = ARM_CONTROL$CONV_WINDOW,
          TOL = ARM_CONTROL$TOL
        )
        return(chain.id)
      },
      cl = NCORES
    )
  }

  res$data_list <- data
  res$chains <- chains
  res$time <- difftime(Sys.time(), start, units = 'secs')
  res$pars <- purrr::reduce(purrr::map(chains, ~ .$avdelta), rbind)
  if (VERBOSE) {
    cat("Done!\n")
  }

  return(res)
}
