data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  real y[n]
}

transformed data {
  real m0 = 10; 
  real slab_scale = 3;
  real slab_scale2 = square(slab_scale);
  real slab_df = 25;
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  vector[p] beta_t;
  vector<lower=0>[p] lambda;
  real<lower=0> c_t;
  real<lower=0> tau_t;
  real alpha;
  real<lower=0> sigma;
}

transformed parameters {
  vector[M] beta;
  {
    real tau0 = (m0 / (M - m0)) * (sigma / sqrt(1.0 * N));
    real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)
    real c2 = slab_scale2 * c2_tilde;

    vector[p] lambda_tilde =
      sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );    
    beta = tau * lambda_tilde .* beta_tilde;
  }
}

model {
  beta_t ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_t ~ cauchy(0, 1);
  c_t ~ inv_gamma(half_slab_df, half_slab_df);

  alpha ~ normal(0, 2);
  sigma ~ normal(0, 2);

  y ~ normal(X * beta + alpha, sigma);
}
