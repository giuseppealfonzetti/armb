##### minus loglikelihood
#' @export
loglik_agh <- function(param, data) {
  p <- length(param) - 1
  beta <- param[1:p]
  sigma <- exp(param[p + 1])
  logl  <- likAGH(beta, sigma, data$x, data$y, data$den, data$niter, data$ws, data$z)
  return(-logl)
}


##### minus  gradient
#' @export
grad_agh <- function(param, data) {
  p <- length(param) - 1
  beta <- param[1:p]
  sigma <- exp(param[p + 1])
  g  <- grAGH(beta, sigma, data$x, data$y, data$den, data$niter, data$ws, data$z)
  g[p + 1] <- g[p + 1] * sigma
  return(-g)
}

