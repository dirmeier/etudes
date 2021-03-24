functions {
  vector f_star_rng(vector f, real[] x, real[] x_star, real alpha, real rho) {
    int n = size(x);
    int n_star = size(x_star);
    matrix[n, n] K = cov_exp_quad(x, alpha, rho)
        + diag_matrix(rep_vector(1e-10, n));
    matrix[n, n] L_K = cholesky_decompose(K);

    matrix[n_star, n] K_star = cov_exp_quad(x_star, x, alpha, rho);
    matrix[n_star, n_star] K_star_star = cov_exp_quad(x_star, alpha, rho);
    matrix[n, n_star] A_star = mdivide_left_tri_low(L_K, K_star');

    vector[n_star] f_star_mean = K_star * inverse(K) * f;
    matrix[n_star, n_star] f_star_cov =  K_star_star
        - A_star' * A_star
        + diag_matrix(rep_vector(1e-10, n_star));

    vector[n_star] f_star = multi_normal_cholesky_rng(
        f_star_mean,
        f_star_cov
    );
    return f_star;
  }
}

data {
  int<lower=1> n;
  real x[n];

  int<lower=1> n_star;
  real x_star[n_star];

  vector[n] f;
  real<lower=0> alpha;
  real<lower=0> rho;
}

generated quantities {
  vector[n_star] f_star;
  f_star = f_star_rng(f, x, x_star, alpha, rho);
}
