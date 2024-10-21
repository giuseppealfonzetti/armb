set.seed(123)
n <-  100
p <- 10
beta <- rnorm(p)
xmat <- matrix(rnorm(n*p), n, p)

response_types <- list(binomial(link = "logit"), poisson(link = "log"))

for (resp in response_types) {
  y <- as.numeric(simy(FAMILY = resp, ETA = xmat%*%beta))

  tmp <- test_glm(y, xmat, resp$family, resp$link, beta)

  test_that(paste0(resp$family, " ", resp$link,": test linkinv"), {

    expect_equal(
      tmp$mu,
      as.numeric(resp$linkinv(xmat%*%beta))
    )
  })

  test_that(paste0(resp$family, " ", resp$link,": test deviance"), {

    expect_equal(
      tmp$dev,
      sum(resp$dev.resids(y, resp$linkinv(xmat%*%beta), 1))
    )
  })

  RFUN <- function(PAR){
    test_glm(y, xmat, resp$family, resp$link, PAR)$nll
  }


  test_that(paste0(resp$family, " ", resp$link,": test gradient computation"), {
    expect_equal(
      test_glm(y, xmat, resp$family, resp$link, beta)$ngr,
      as.numeric(t(xmat)%*%(resp$linkinv(xmat%*%beta)-y))/n
    )
  })

  test_that(paste0(resp$family, " ", resp$link,": test gradient computation vs numerical"), {
    skip_if_not_installed("numDeriv")

    expect_equal(
      as.numeric(t(xmat)%*%(resp$linkinv(xmat%*%beta)-y))/n,
      numDeriv::grad(RFUN, beta),
      tolerance = 1e-5
    )
  })

  test_that(paste0(resp$family, " ", resp$link,": test numerical gradient"), {
    skip_if_not_installed("numDeriv")
    expect_equal(
      test_glm(y, xmat, resp$family, resp$link, beta)$ngr,
      numDeriv::grad(RFUN, beta),
      tolerance = 1e-5
    )
  })
}

