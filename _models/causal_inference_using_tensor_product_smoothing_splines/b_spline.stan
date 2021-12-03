#include functions.stan

data {
  int<lower=0> N;
  vector[N] X;
  vector[N] y;

  int<lower=0> degree;
  int<lower=0> n_knots;
  vector[n_knots] knots;
}

transformed data {
  int n_coef = n_knots - degree - 1;
  matrix[N, n_coef] B = bspline(X, n_coef, degree, knots);
}


parameters {
  vector[n_coef] mu;
  real<lower=0> mu_tau;
  real<lower=0> sigma;
}

model {
  mu_tau ~ inv_gamma(4, 4);
  mu[1] ~ std_normal();
  for (i in 2:n_coef) {
      mu[i] ~ normal(mu[i - 1], mu_tau);
  }

  y ~ normal(B * mu, sigma);
}

generated quantities {
  vector[N] y_hat;
  y_hat = B * mu;
}


