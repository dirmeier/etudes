data {
  int<lower=0> N;
  
  vector[N] x;
  vector[N] y;

  real<lower=0> lambda;
}

parameters {
  real beta;
  real sigma;
  real tau;
}

model {
  tau ~ exponential(lambda  * lambda / 2);
  sigma ~ inv_gamma(1, 1);
  beta ~ normal(0, sigma * tau);
  
  y ~ normal(beta * x, sigma);
}
