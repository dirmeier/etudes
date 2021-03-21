functions {
 vector reparam(vector f_tilde, real[] x, real alpha, real rho)  {
    int n = num_elements(x);
    vector[n] f;
    matrix[n, n] L_K;
    matrix[n, n] K = cov_exp_quad(x, alpha, rho)
        + diag_matrix(rep_vector(1e-10, n));
    L_K = cholesky_decompose(K);
    f = L_K * f_tilde;
    return f;
 }
}

data {
  int<lower=1> n;
  real x[n];
  int y[n];

  int<lower=1> n_star;
  real x_star[n_star];
}


parameters {
  vector[n] f_tilde;
  real<lower=0> alpha;
  real<lower=0> rho;
}

model {
  vector[n] f = reparam(f_tilde, x, alpha, rho);
  f_tilde ~ std_normal();
  rho ~ inv_gamma(25, 5);
  alpha ~ normal(0, 2);

  y ~ bernoulli_logit(f);
}
