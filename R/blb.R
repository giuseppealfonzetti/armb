#' @export
blb <- function(
  data,
  subset_size_b = nrow(data)^0.7,
  n_subsets = NA,
  n_resamples = 100,
  window_subsets = 3,
  window_resamples = 20,
  epsilon = 0.05,
  fun_estimator = NULL,
  fun_metric = NULL,
  verbose = TRUE,
  seed = 123
) {
  stopifnot(
    is.function(fun_estimator),
    is.function(fun_metric)
  )

  # size of data
  n <- nrow(data)

  # disjoint random subsets of the observed data each of
  # approximate size `subset_size_b`
  set.seed(seed)
  ids_subsets <- sample(
    x = ceiling(n / subset_size_b),
    size = n,
    replace = TRUE
  )

  # number of sampled subsets
  n_subsets <- if (is.na(n_subsets)) {
    max(ids_subsets)
  } else {
    min(n_subsets, max(ids_subsets))
  }

  # number of resamples per subset
  ids_r <- seq_len(n_resamples)

  if (verbose) {
    cat(
      "sample size:",
      n,
      ", n subsets:",
      n_subsets,
      ", R per subset:",
      n_resamples,
      "\n"
    )
  }

  # containers for estimators of interests and of quality assessments
  out_quality <- names(fun_metric(stats::runif(10)))
  ntmp <- seq_len(min(20, n))
  out_estimators <- names(suppressWarnings(fun_estimator(data[ntmp, ], ntmp)))

  res_ests <- array(
    NA,
    dim = c(n_subsets, length(out_estimators), n_resamples),
    dimnames = list(NULL, out_estimators, NULL)
  )
  res_qual <- array(
    NA,
    dim = c(n_subsets, length(out_quality), length(out_estimators)),
    dimnames = list(NULL, out_quality, out_estimators)
  )

  # loop over each of the s subsets
  for (j in seq_len(n_subsets)) {
    ids_j <- which(ids_subsets == j)
    b <- length(ids_j)

    if (verbose) {
      cat("\nSubset:", j, ", subset size:", b, "\n")
    }

    # generate the r Monte Carlo resamples
    resample_weights <- stats::rmultinom(
      n = n_resamples,
      size = n,
      prob = rep(1, b)
    )

    # loop over each of the r Monte Carlo resamples/iterations
    for (k in ids_r) {
      # calculate estimator(s) of interest
      res_ests[j, , k] <- suppressWarnings(
        fun_estimator(data[ids_j, ], weights = resample_weights[, k])
      )

      # if estimator(s) of interest have converged then stop resamples
      if (k > window_resamples) {
        z_t <- res_ests[j, , k]
        tmp <- sweep(
          x = res_ests[j, , seq(k - window_resamples, k - 1)],
          MARGIN = 1,
          STATS = z_t,
          FUN = "-"
        )
        tmp <- sweep(abs(tmp), 1, abs(z_t), FUN = "/")
        have_resamples_conv <- all(colMeans(tmp) < epsilon)

        if (have_resamples_conv) break
      }
    }

    if (!have_resamples_conv) {
      warning(
        "Subset ",
        j,
        " has not converged: ",
        "number of resamples (r = ",
        n_resamples,
        ") may be too small."
      )
    }

    # calculate estimator(s) quality assessment for subset j across resamples
    res_qual[j, , ] <- apply(
      X = res_ests[j, , seq_len(k), drop = FALSE],
      MARGIN = 2,
      FUN = fun_metric
    )

    # if estimator(s) quality assessment have converged then stop subsets
    if (j > window_subsets) {
      z_t <- res_qual[j, , , drop = FALSE]
      tmp <- sweep(
        x = res_qual[seq(j - window_subsets, j - 1), , , drop = FALSE],
        MARGIN = 2:3,
        STATS = z_t,
        FUN = "-"
      )
      tmp <- sweep(abs(tmp), 2:3, abs(z_t), FUN = "/")
      have_subsets_conv <- all(rowMeans(tmp) < epsilon)

      if (have_subsets_conv) break
    }
  }

  if (!have_subsets_conv) {
    warning(
      "BLB has not converged: ",
      "number of subsets (s = ",
      n_subsets,
      ") may be too small."
    )
  }

  # Average of quality assessment estimates across the s subsets
  colMeans(res_qual[seq_len(j), , , drop = FALSE])
}
