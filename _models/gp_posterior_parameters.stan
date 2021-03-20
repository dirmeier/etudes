data {
  int<lower=1> n;
  real x[n];
  vector[n] y;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
  real<lower=0> sigma;

  vector[n] f_tilde;
}

model {
  vector[n] f;
  {
    matrix[n, n] K =  cov_exp_quad(x, alpha, rho)
        + diag_matrix(rep_vector(1e-10, n));
    matrix[n, n] L_K = cholesky_decompose(K);
    f = L_K * f_tilde;
  }

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ std_normal();
  f_tilde ~ std_normal();

  y ~ normal(f, sigma);
}
