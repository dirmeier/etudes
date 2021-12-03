#include functions.stan

data {
  int<lower=0> N;
  vector[2] X[N];
  vector[N] y;

  int<lower=0> degree;
  int<lower=0> n_knots;
  vector[n_knots] x1_knots;
  vector[n_knots] x2_knots;
}

transformed data {
  int n_coef = n_knots - degree - 1;

  vector[n_knots] knots[2];
  knots[1] = x1_knots;
  knots[2] = x2_knots;

  matrix[N, n_coef * n_coef] B = tensor_spline(X, n_coef, degree, knots);
}

parameters {
  vector[n_coef  * n_coef] mu;
  real<lower=0> mu_tau;
  real<lower=0> sigma;
}

model {
  mu_tau ~ inv_gamma(4, 4);
  mu[1] ~ std_normal();
  for (i in 2:(n_coef  * n_coef)) {
      mu[i] ~ normal(mu[i - 1], mu_tau);
  }

  y ~ normal(B * mu, sigma);
}

generated quantities {
  vector[N] y_hat;
  y_hat = B * mu;
}


