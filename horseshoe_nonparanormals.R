library(Matrix)
library(MASS)
library(rstan)

p <- 5
n <- 100

set.seed(1)
A <- rsparsematrix(p, p, 0.15, rand.x = rnorm)
Lambda <- A %*% t(A) + 0.05 * diag(p)

Sigma <- as.matrix(solve(Lambda))

Y <- MASS::mvrnorm(n, rep(0, p), Sigma)

fit <- rstan::stan(
  "_models/horseshoe_nonparanormals.stan",
  iter = 3000,
  warmup = 2000,
  data=list(N=nrow(Y),
            P=ncol(Y),
            P_upper=as.integer(p * (p - 1) / 2),
            y=Y),
  seed=23,
  control=list(adapt_delta=0.99, max_treedepth=15)
)
fit

omega <- extract(fit)$omega
omega <- apply(omega, 2, mean)
omega_diag <- extract(fit)$omega_diag
omega_diag <- apply(omega_diag, 2, mean)

Omega <- matrix(0, p, p)
Omega[lower.tri(Omega)]  <- omega
Omega <- t(Omega) + Omega
diag(Omega) <-omega_diag
Omega[abs(Omega) < 0.001] <- 0

Omega
Lambda
