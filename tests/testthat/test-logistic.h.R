n <-  5
m <- 4

set.seed(123)
x <- cbind(1,matrix(rnorm(n*(m-1)), n, m-1))
beta <- rnorm(m)

eta <- x %*% beta
mu <- 1/(1 + exp(-eta))
y <- sapply(mu, FUN = function(x){rbinom(1,1,prob = x)})
y

R_nll <- function(PAR)
{
  logistic_wrapper(Y = as.numeric(y), X = x, THETA = PAR)$nll
}

R_ngr <- function(PAR)
{
  logistic_wrapper(Y = as.numeric(y), X = x, THETA = PAR)$ngr
}

test_that("check gradient" , {
  skip_if_not_installed("numDeriv")
  expect_equal(numDeriv::grad(R_nll, beta), R_ngr(beta))
})
