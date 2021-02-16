
data {
  int<lower=1> N;
  real x[N];
  int y[N];
  real<lower=0> alpha;
  real<lower=0> rho;
}

transformed data {
  real delta = 1e-9;
}

parameters {
  vector[N] f_tilde;
}

model {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x, alpha, rho);

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + delta;

    L_K = cholesky_decompose(K);
    f = L_K * f_tilde;
  }

  f_tilde ~ std_normal();

  y ~ bernoulli_logit(f);
}

