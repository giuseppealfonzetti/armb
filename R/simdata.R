#' @export
simy <- function(FAMILY, ETA){
  mu <- FAMILY$linkinv(ETA)

  out <- switch(FAMILY$family,
                "binomial" = sapply(mu, function(mui) stats::rbinom(1,1,mui)),
                "poisson"  = sapply(mu, function(mui) stats::rpois(1,mui)))
  return(out)
}
