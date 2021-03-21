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

  vector pred_f_star(vector f_tilde, real[] x, real[] x_star, real alpha, real rho) {
    int n = size(x);
    int n_star = size(x_star);
    matrix[n, n] K = cov_exp_quad(x, alpha, rho);
    matrix[n, n_star] K_star = cov_exp_quad(x, x_star, alpha, rho);
    vector[n] f_post = reparam(f_tilde, x, alpha, rho);
    vector[n_star] f_star = K_star' * inverse(K) * f_post;
    return f_star;
  }
}

data {
  int<lower=1> n;
  real x[n];

  int<lower=1> n_star;
  real x_star[n];

  vector[n] f_tilde;
  real<lower=0> alpha;
  real<lower=0> rho;
}

generated quantities {
  vector[n_star] f_star;
  f_star = pred_f_star(f_tilde, x, x_star, alpha, rho);
}
